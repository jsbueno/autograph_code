import os, sys

import threading, time


import bpy
from bpy.props import StringProperty, PointerProperty, BoolProperty

from bpy.types import Panel, Operator, PropertyGroup


def autograph_path():
    """Enable 3rd party Python modules installed in an
    active Virtualenv when Python is run.
    (Blender ignores the virtualenv, and uses a
    Python binary hardcoded in the build. This works
    if the virtualenv's Python is the same as blenders')
    """

    import sys, os
    from pathlib import Path

    pyversion_path = f"python{sys.version_info.major}.{sys.version_info.minor}"

    for pathcomp in os.environ["PATH"].split(os.pathsep)[::-1]:
        p = Path(pathcomp)
        if p.name != "bin":
            continue
        lib_path = p.parent / "lib" / pyversion_path / "site-packages"
        if (not lib_path.exists()) or str(lib_path) in sys.path:
            continue
        sys.path.insert(0, str(lib_path))

        print(f"Autograph extension: prepended {lib_path!r} to PYTHONPATH")


autograph_path()

try:
    import pyautogui
except ImportError:
    print("Could not import Pyautogui - some niceties may not work", file=sys.stderr)
    pyautogui = None



TEMP_ACTION_ID = "temp_action"
AUTOGRAPH_ID = "Autograph"


bl_info = {
    "name": "Autograph",
    "category": "Object",
}

def scene_cleanup(context):

    for grease in bpy.data.grease_pencil:
        bpy.data.grease_pencil.remove(grease)

    for obj in bpy.data.objects:
        if obj.name.startswith("GP_Layer"):
            bpy.data.objects.remove(obj)

    anim_data = bpy.data.objects[AUTOGRAPH_ID].animation_data

    for track in anim_data.nla_tracks:
        anim_data.nla_tracks.remove(track)

    # Trigger for screen update
    bpy.data.objects[0].location.x += 0.0

    for action in bpy.data.actions:
        if action.name.startswith("temp_"):
            bpy.data.actions.remove(action)

    press_and_hold_grease_pencil_key(15)


def press_and_hold_grease_pencil_key(timeout=15):
    if not pyautogui:
        print("No autogui")
        return
    def hold_key():
        print("pressing 'd'")
        pyautogui.keyDown("d")
        time.sleep(timeout)
        pyautogui.keyUp("d")

    t = threading.Thread(target=hold_key)
    t.start()


def cleanup_speeds(v):
    new_v = []
    started = False
    stroke = []
    for value in v:
        if not started and value < 0.01:
            continue
        started = True
        if value > 0.01:
            stroke.append(value)
        else:
            if stroke:
                new_v.append(stroke)
                stroke = []
    return new_v


def autograph(context):

    strokes = bpy.data.grease_pencil[0].layers[0].active_frame.strokes

    velocidades = []
    pressoes = []
    pontos = []

    for word, stroke in strokes.items():
        velocidade = []; pressao = []
        for i, point in stroke.points.items():
            if i == 0:
                previous = point
                continue
            pressao.append(point.pressure)
            velocidade.append((point.co - previous.co).magnitude)
            previous = point
        if i <= 1:
            continue
        velocidades.append(sum(velocidade) / len(velocidade))
        pressoes.append(sum(pressao) / len(pressao))
        pontos.append(i)

    bpy.ops.gpencil.convert(type='POLY', use_timing_data=True)

    anim_curve = bpy.data.curves[0].animation_data.action.fcurves[0]
    points = bpy.data.curves[0].splines[0].points

    anim_curve.convert_to_samples(0, 20000)
    velocidades_curvas = []
    prev = None
    for point in anim_curve.sampled_points:
        if prev == None:
            prev = point
            continue
        velocidade = (point.co - prev.co).y
        velocidades_curvas.append(velocidade)
        prev = point

    velocidades_curvas = cleanup_speeds(velocidades_curvas)

    print(f"Velocidades a partir das curvas: {velocidades_curvas}")

    # anum_curve = bpy.data.curves[0].animation_data.action.fcurves[0].sampled_points[i].co


    print(f"velocidades={velocidades}\npressoes={pressoes}")


def concatenate_action(action, previous, ignore_height=True):
    """
    Changes target action curves by translating and rotating the
    root bone curves so that it continues smoothly from the point
    where the previous action had stopped.

    We do not do that for the "y" curve - the "height to the ground" for
    the root base by default, otherwise the character could
    "sink into the ground" after an action that ends in a lower plane

    """
    print(f"concatenating actions {previous.name} and {action.name}")

    curve_indexes = [0, 1, 2, 3, 4, 5]
    # rotation_indexes = [3, 4, 5]
    if ignore_height:
        curve_indexes.remove(1)

    prev_start, prev_end = previous.fcurves[0].range()
    start, end = action.fcurves[0].range()

    base_values = []
    zero_values = []
    for index in curve_indexes:
        previous.fcurves[index].convert_to_samples(prev_start, prev_end)
        base_values.append(previous.fcurves[index].sampled_points[int(prev_end) - 2].co[1])

        action.fcurves[index].convert_to_samples(start, end)
        zero_values.append(action.fcurves[index].sampled_points[int(start)].co[1])

    for index, base_value, zero_value in zip(curve_indexes, base_values, zero_values):
        curve = action.fcurves[index]
        for point in curve.sampled_points:
            value = point.co[1] + base_value - zero_value

            # descanbalhota:
            #if "rotation_euler" in curve.data_path:
                #value %= 360
            point.co[1] = value


def assemble_actions(context, action_list):


    autograph = bpy.data.objects[AUTOGRAPH_ID]

    track_name =  AUTOGRAPH_ID.lower() + "_dance"
    try:
        old_track = autograph.animation_data.nla_tracks[track_name]
    except KeyError:
        pass
    else:
        autograph.animation_data.nla_tracks.remove(old_track)

    track = autograph.animation_data.nla_tracks.new()
    track.name = track_name

    previous_end = 0
    prev_action = None
    for action_name in action_list:

        try:
            action = bpy.data.actions[action_name]
        except Exception as error:
            print(f"encrencou a action {action_name}")
            continue
        new_action = action.copy()
        new_action.name = "temp_" + action_name

        #if prev_action:
            #concatenate_action(new_action, prev_action)
        strip = track.strips.new(action.name, previous_end, new_action)
        strip.select = False
        previous_end += action.frame_range[1] + 15
        prev_action = new_action

    previous_strip = None

    context.scene.objects.active = autograph

    # make armature visible so that it is added to the NLA
    context.scene.layers[10] = True

    # autograph.animation_data_clear()

    original_area = context.area.type
    context.area.type = "NLA_EDITOR"
    bpy.ops.nla.selected_objects_add()
    track.select = True
    bpy.ops.nla.select_all_toggle(True)
    bpy.ops.nla.transition_add()

    context.area.type = original_area

    # Hide armature to play animation:
    context.scene.layers[10] = False

    return previous_end - 15

    # bpy.data.objects[AUTOGRAPH_ID].animation_data.action = TEMP_ACTION_ID



def autograph_test(context):
    total_frames = assemble_actions(
        context,
        "Bia_T_1;r_m;Amanda_A_1;Bia_B_1__001.001;Samara_A_1;Helder_L_1;Helder_H_2;Thiago_O_3".split(";")
    )  #["S_f", "r_f"])
    context.scene.frame_end = total_frames

    bpy.ops.screen.animation_play()


class Autograph(Operator):
    """Launch Dance Action"""

    bl_idname = "add.autograph"
    bl_label = "Autograph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        autograph(context)
        return {'FINISHED'}

class AutographTest(Operator):
    """Launch Dance for Single Letter"""

    bl_idname = "test.autograph"
    bl_label = "TESTE"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        autograph_test(context)
        return {'FINISHED'}


class AutographClear(Operator):
    """Clear Grease Writting"""

    bl_idname = "clear.autograph"
    bl_label = "Limpar escrita"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        scene_cleanup(context)
        return {'FINISHED'}



class AutographPanel(Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Autograph"
    bl_idname = "AUTOGRAPH_part1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_category = "Create"
    bl_context = "objectmode"

    def draw(self, context):
        print("autograph")
        layout = self.layout

        scene = context.scene

        layout.label(text="AUTOGRAPH")
        row = layout.row()
        row.operator("add.autograph")
        row = layout.row()
        row.operator("clear.autograph")
        row = layout.row()
        row.operator("test.autograph")


def register():
    print("Resgistering Autograph add-on")
    bpy.utils.register_module(__name__)
    # bpy.types.Scene.autograph = PointerProperty(type=AutoSettings)


def unregister():
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
