import os, sys
import random
import threading, time

from contextlib import contextmanager

import bpy
from bpy.props import StringProperty, PointerProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup



AUTOGRAPH_PHRASE = "escrevercomocorpo"

START_WRITTING_TIMEOUT = 15
STOPPED_WRITTING_TIMEOUT = 3
TEMP_ACTION_ID = "temp_action"
AUTOGRAPH_ID = "Autograph_Skel"

WRITTING_COLOR = (0, 0.1, 0)
POST_WRITTING_COLOR = (0, 0, 0.2)

ARMATURE_LAYER = 1


ACTION_DATA = {
 'a': [{'name': 'amanda_a_1'},
  {'name': 'amanda_a_2'},
  {'name': 'amanda_a_3'},
  {'name': 'amanda_a_4'},
  {'name': 'bia_a_1'},
  {'name': 'jessica_a_1'},
  {'name': 'jessica_a_2'},
  {'name': 'jessica_a_3'},
  {'name': 'jessica_a_4'},
  {'name': 'samara_a_1'},
  {'name': 'samara_a_2'},
  {'name': 'samara_a_3'},
  {'name': 'samara_a_4'},
  {'name': 'thiago_a_1'},
  {'name': 'thiago_a_2'},
  {'name': 'thiago_a_3'},
  {'name': 'thiago_a_4'},
  {'name': 'tiago_a_1'},
  {'name': 'tiago_a_2'},
  {'name': 'tiago_a_3'},
  {'name': 'tiago_a_4'},
  {'name': 'tiago_a_5'},
  {'name': 'tiago_a_6'}],
 'aa': [{'name': 'samara_aa_1'},
  {'name': 'samara_aa_2'},
  {'name': 'samara_aa_3'},
  {'name': 'samara_aa_4'}],
 'aaa': [{'name': 'samara_aaa_1'},
  {'name': 'samara_aaa_2'},
  {'name': 'samara_aaa_3'},
  {'name': 'samara_aaa_4'}],
 'b': [{'name': 'bia_b_1'}],
 'c': [{'name': 'jessica_c_1'},
  {'name': 'jessica_c_2'},
  {'name': 'jessica_c_3'},
  {'name': 'jessica_c_4'},
  {'name': 'jessica_c_5'},
  {'name': 'jessica_c_6'}],
 'd': [{'name': 'diego_d_1'},
  {'name': 'helder_d_1'},
  {'name': 'helder_d_3'},
  {'name': 'helder_d_4'}],
 'e': [{'name': 'bia_e_1'},
  {'name': 'diego_e_1'},
  {'name': 'helder_e_1'},
  {'name': 'helder_e_2'},
  {'name': 'helder_e_3'},
  {'name': 'helder_e_4'},
  {'name': 'jessica_e_1'},
  {'name': 'jessica_e_2'},
  {'name': 'jessica_e_3'},
  {'name': 'jessica_e_4'},
  {'name': 'jessica_e_5'},
  {'name': 'jessica_e_6'},
  {'name': 'jessica_e_7'}],
 'g': [{'name': 'diego_g_1'},
  {'name': 'thiago_g_1'},
  {'name': 'thiago_g_2'},
  {'name': 'thiago_g_3'},
  {'name': 'thiago_g_4'},
  {'name': 'tiago_g_1'},
  {'name': 'tiago_g_2'},
  {'name': 'tiago_g_3'},
  {'name': 'tiago_g_4'}],
 'h': [{'name': 'helder_h_1'},
  {'name': 'helder_h_2'},
  {'name': 'helder_h_3'},
  {'name': 'thiago_h_1'},
  {'name': 'thiago_h_2'},
  {'name': 'thiago_h_3'},
  {'name': 'thiago_h_4'},
  {'name': 'thiago_h_5'}],
 'i': [{'name': 'bia_i_1'},
  {'name': 'diego_i_1'},
  {'name': 'jessica_i_1'},
  {'name': 'jessica_i_2'},
  {'name': 'jessica_i_3'},
  {'name': 'jessica_i_4'},
  {'name': 'jessica_i_5'},
  {'name': 'thiago_i_1'},
  {'name': 'thiago_i_2'},
  {'name': 'thiago_i_3'},
  {'name': 'thiago_i_4'},
  {'name': 'tiago_i_1'}],
 'j': [{'name': 'jessi_j_2'},
  {'name': 'jessica_j_'},
  {'name': 'jessica_j_1'},
  {'name': 'jessica_j_2'},
  {'name': 'jessica_j_3'},
  {'name': 'jessica_j_4'},
  {'name': 'jessica_j_5'}],
 'l': [{'name': 'helder_l_1'},
  {'name': 'helder_l_2'},
  {'name': 'helder_l_3'},
  {'name': 'helder_l_4'}],
 'm': [{'name': 'amanda_m_'},
  {'name': 'amanda_m_1'},
  {'name': 'amanda_m_2'},
  {'name': 'amanda_m_3'},
  {'name': 'samara_m_1'},
  {'name': 'samara_m_2'},
  {'name': 'samara_m_4'},
  {'name': 'samra_m_3'}],
 'n': [{'name': 'amanda_n_1'},
  {'name': 'amanda_n_2'},
  {'name': 'amanda_n_3'},
  {'name': 'amanda_n_4'},
  {'name': 'amanda_n_5'},
  {'name': 'amanda_n_6'},
  {'name': 'amanda_n_7'},
  {'name': 'amanda_n_8'}],
 'o': [{'name': 'diego_o_1'},
  {'name': 'thiago_o_1'},
  {'name': 'thiago_o_2'},
  {'name': 'thiago_o_3'},
  {'name': 'thiago_o_4'}],
 'p': [{'name': 'bia_r_1'},
  {'name': 'helder_r_1'},
  {'name': 'helder_r_2'},
  {'name': 'helder_r_3'},
  {'name': 'helder_r_4'},
  {'name': 'samara_r_1'},
  {'name': 'samara_r_2'},
  {'name': 'samara_r_3'},
  {'name': 'samara_r_4'}],
 'r': [{'name': 'bia_r_1'},
  {'name': 'helder_r_1'},
  {'name': 'helder_r_2'},
  {'name': 'helder_r_3'},
  {'name': 'helder_r_4'},
  {'name': 'samara_r_1'},
  {'name': 'samara_r_2'},
  {'name': 'samara_r_3'},
  {'name': 'samara_r_4'}],
 's': [{'name': 'jessica_s_1'},
  {'name': 'jessica_s_2'},
  {'name': 'jessica_s_3'},
  {'name': 'jessica_s_4'},
  {'name': 'jessica_s_5'},
  {'name': 'jessica_s_6'},
  {'name': 'samara_s_1'},
  {'name': 'samara_s_2'},
  {'name': 'samara_s_3'},
  {'name': 'samara_s_4'},
  {'name': 'thiago_s_1'},
  {'name': 'thiago_s_2'},
  {'name': 'thiago_s_3'},
  {'name': 'thiago_s_4'},
  {'name': 'thiago_s_5'},
  {'name': 'thiago_s_6'}],
 't': [{'name': 'bia_t_1'},
  {'name': 'thiago_t_1'},
  {'name': 'thiago_t_2'},
  {'name': 'thiago_t_3'},
  {'name': 'thiago_t_4'}],
 'v': [{'name': 'helder_w_1'},
  {'name': 'helder_w_2'},
  {'name': 'helder_w_3'},
  {'name': 'helder_w_4'},
  {'name': 'helder_w_5'}],
 'w': [{'name': 'helder_w_1'},
  {'name': 'helder_w_2'},
  {'name': 'helder_w_3'},
  {'name': 'helder_w_4'},
  {'name': 'helder_w_5'}],
 'z': [{'name': 'bia_z_1'}]}



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


bl_info = {
    "name": "Autograph",
    # "location": "View3D > Tools > Autograph"
    "category": "Autograph",
    "author": "João S. O. Bueno",
    "version": (0, 2, 0)
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

    bpy.ops.screen.animation_cancel()
    bpy.context.scene.frame_current = 0
    # bpy.data.grease_pencil["GPencil"].palettes["GP_Palette"].colors["Color"].color = WRITTING_COLOR


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


@contextmanager
def switch_context_area(context, area="NLA_EDITOR"):
    original_area = context.area.type
    context.area.type = area
    yield
    context.area.type = original_area


@contextmanager
def activate_layer(context, layer):
    original_value = context.scene.layers[layer]
    context.scene.layers[layer] = True
    yield
    context.scene.layers[layer] = original_value


def autograph(context):
    """
    Main functionality -
    this will extract writting metrics from the active grease-pencil
    layer, and use those to select the proper actions to be
    associated with the main Autograph actor.
    """

    try:
        strokes = bpy.data.grease_pencil[0].layers[0].active_frame.strokes
    except IndexError:
        print("Can't start: No grease pencil writting found on scene.")
        return

    speed_per_stroke = []
    pressure_per_stroke = []
    points = []

    for word, stroke in strokes.items():
        speed = []; pressao = []
        for i, point in stroke.points.items():
            if i == 0:
                previous = point
                continue
            pressao.append(point.pressure)
            speed.append((point.co - previous.co).magnitude)
            previous = point
        if i <= 1:
            continue
        speed_per_stroke.append(sum(speed) / len(speed))
        pressure_per_stroke.append(sum(pressao) / len(pressao))
        points.append(i)

    bpy.ops.gpencil.convert(type='POLY', use_timing_data=True)

    anim_curve = bpy.data.curves[0].animation_data.action.fcurves[0]
    points = bpy.data.curves[0].splines[0].points

    anim_curve.convert_to_samples(0, 20000)
    speed_per_stroke_curves = []
    prev = None
    for point in anim_curve.sampled_points:
        if prev == None:
            prev = point
            continue
        speed = (point.co - prev.co).y
        speed_per_stroke_curves.append(speed)
        prev = point

    speed_per_stroke_curves = cleanup_speeds(speed_per_stroke_curves)

    print(f"Speeds measured from curves: {speed_per_stroke_curves}")

    # anum_curve = bpy.data.curves[0].animation_data.action.fcurves[0].sampled_points[i].co


    print(f"speed_per_stroke={speed_per_stroke}\npressure_per_stroke={pressure_per_stroke}")


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


def get_action_names(phrase):
    actions = []
    for letter in phrase:
        action = random.choice(ACTION_DATA[letter])["name"]
        actions.append(action)
    return actions


def assemble_actions(context, phrase):

    action_list = get_action_names(phrase)

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
            print(f"Expected action not found {action_name!r}")
            continue
        new_action = action.copy()
        new_action.name = "temp_" + action_name

        strip = track.strips.new(action.name, previous_end, new_action)
        strip.select = False
        previous_end += action.frame_range[1] + 15
        prev_action = new_action

    previous_strip = None

    context.scene.objects.active = autograph

    with switch_context_area(context, "NLA_EDITOR"), activate_layer(context, ARMATURE_LAYER):
        bpy.ops.nla.selected_objects_add()
        track.select = True
        bpy.ops.nla.select_all_toggle(True)
        bpy.ops.nla.transition_add()

    return previous_end - 15



def autograph_test(context):
    total_frames = assemble_actions(context, AUTOGRAPH_PHRASE)
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

    _timer = None

    def execute(self, context):
        scene_cleanup(context)
        self.writting_started = False
        self.modal_func = self.check_writting_started

        self._timer = context.window_manager.event_timer_add(0.2, context.window)
        context.window_manager.modal_handler_add(self)

        self.press_and_hold_grease_pencil_key(START_WRITTING_TIMEOUT)
        return {'RUNNING_MODAL'}


    def press_and_hold_grease_pencil_key(self, timeout=15):
        self.writting_started = False
        if not pyautogui:
            print("No autogui")
            return
        def hold_key():
            pyautogui.keyDown("t")
            time.sleep(0.05)
            pyautogui.keyUp("t")
            start_time = time.time()
            print("pressing 'd'")
            pyautogui.keyDown("d")
            try:
                while time.time() - start_time < timeout:
                    time.sleep(0.2)
                    if self.writting_started:
                        break
            except ReferenceError:
                pass
            finally:
                pyautogui.keyUp("d")

        t = threading.Thread(target=hold_key)
        t.start()


    def check_writting_started(self, context):

        try:
            gp_layer = bpy.data.grease_pencil["GPencil"].layers["GP_Layer"]
        except (KeyError, IndexError, AttributeError):
            print("No writting detected")
            return False
        print("Writting started")
        self.writting_started = True
        self.modal_func = self.check_writting_ended
        gp_layer.line_change = 10
        gp_layer.tint_color = POST_WRITTING_COLOR
        gp_layer.tint_factor = 1.0
        bpy.data.grease_pencil["GPencil"].layers["GP_Layer"].line_change = 10

        bpy.data.grease_pencil["GPencil"].palettes["GP_Palette"].colors["Color"].color = WRITTING_COLOR

        self.check_write_strokes = 0
        self.check_write_points = 0
        self.check_write_time = time.time()
        return False

    def check_writting_ended(self, context):
        return False

        #try:
            #strokes = bpy.data.grease_pencil["GPencil"].layers["GP_Layer"].active_frame.strokes
        #except (IndexError, KeyError, AttributeError):
            #print("No writting - something went wrong")
            #return True
        #if len(strokes) and (len(strokes) > self.check_write_strokes or len(strokes[-1].points) > self.check_write_points):
            #self.check_write_time = time.time()
            #self.check_write_strokes = len(strokes)
            #self.check_write_points = len(strokes[-1].points)

            #return False

        #if time.time() - self.check_write_time > STOPPED_WRITTING_TIMEOUT:
            #if pyautogui:
                #pyautogui.press("escape")
                #pyautogui.press("t")

            #return True
        #return False


    def modal(self, context, event):
        if event.type == "TIMER":
            if self.modal_func(context):
                return {"FINISHED"}

        return {"PASS_THROUGH"}


class AutographPanel(Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Autograph v.%d.%d.%d" % bl_info['version']
    bl_idname = "AUTOGRAPH_part1"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_category = "Autograph"
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
    print("Registering Autograph add-on")
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
