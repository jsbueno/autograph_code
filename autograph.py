import os, sys
import random
import threading, time

from collections import namedtuple
from contextlib import contextmanager

import bpy
from bpy.props import StringProperty, PointerProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup

import flipper


AUTOGRAPH_PHRASE = "escrever com o corpo"

SPACE_MARGIN = 2

START_WRITTING_TIMEOUT = 15
STOPPED_WRITTING_TIMEOUT = 6
TEMP_ACTION_ID = "temp_action"
AUTOGRAPH_ID = "Autograph_Skel"

WRITTING_COLOR = (0, 0.1, 0)
POST_WRITTING_COLOR = (0, 0, 0.2)

ARMATURE_LAYER = 2
ACTION_SPACING = 15

ROOT_X_NAME = """pose.bones["pelvis"].location"""

from autograph_action_data import data as ACTION_DATA


def autograph_path():
    """Enable 3rd party Python modules installed in an
    active Virtualenv when Python is run.
    (Blender ignores the virtualenv, and uses a
    Python binary hardcoded in the build. This works
    if the virtualenv's Python is the same as blenders')
    """

    import sys, os
    from pathlib import Path

    pyversion_path = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)

    for pathcomp in os.environ["PATH"].split(os.pathsep)[::-1]:
        p = Path(pathcomp)
        if p.name != "bin":
            continue
        lib_path = p.parent / "lib" / pyversion_path / "site-packages"
        if (not lib_path.exists()) or str(lib_path) in sys.path:
            continue
        sys.path.insert(0, str(lib_path))

        print("Autograph extension: prepended {} to PYTHONPATH".format(lib_path))


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
    "version": (0, 99, 0)
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


def cleanup_and_merge_speeds(v):
    """Removes speed artifacts that take place
    at starting and end of each stroke.

    Also, join all speed data in a plain list, without nested data.
    """
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
                new_v.extend(stroke)
                stroke = []
    if stroke:
        new_v.extend(stroke)
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


def average_value_per_letter(phrase, measured_points, normalize=(0, 1), number_written_letters=len(AUTOGRAPH_PHRASE)):
    """Re-sample measurements according to the number of
    letters expcted in the writting.

    Also normalize measured points so that points at the extremes
    passed are mapped to the 0-1 range.


    Important: samples are taken relative to time of writting -
    so these values are for "neighborhood" of the letter.

    In a stage when we have proper writting recognition
    built-in, the values may be yielded exactly for each glyph.
    """
    phrase = phrase[:number_written_letters]
    trimmed_phrase = phrase.replace(" ", "")
    factor = len(measured_points) / len(trimmed_phrase)
    if factor < 1:
        return measured_points
    new_points = []
    norm_factor = 1 / (normalize[1] - normalize[0])
    for i, letter in enumerate(phrase):
        if letter == " ":
            # for spaces, copy parameters from the previous drawn letter
            new_points.append(new_points[-1] if new_points else 0)
            continue
        points_value = sum(measured_points[int(i * factor): int((i + 1) * factor)]) / factor
        points_value = (points_value - normalize[0]) * norm_factor
        new_points.append(points_value)
    return new_points


def guess_written_phrase_size(strokes, speed):

    phrase = AUTOGRAPH_PHRASE

    average_points_per_letter = 42
    minimal_points_per_letter = 22

    num_words = len(AUTOGRAPH_PHRASE.split())
    total_points = sum(len(stroke.points) for stroke in strokes)
    result = -1
    if len(strokes) < num_words:
        # probably the phrase was truncated
        # use average of 40 points on strokes per letter - ignore speed for this guess.
        result = total_points // average_points_per_letter

    elif num_words < len(strokes) < num_words + 3:
        # cursive text - (not one stroke per letter)
        words = iter(phrase.split())
        word = next(words, "")
        for stroke in strokes:
            if len(stroke.points) / minimal_points_per_letter >= len(word):
                word = next(words, "")
        word = next(words, "")
        if word:
            # strokes not long enough to draw all words in the phrase -> predict truncated size
            result = total_points // average_points_per_letter

    else:
        # a lot of strokes - either user tried to imitate printing types (letra de forma)
        # or we just have a mess of points and traces.
        if total_points <= minimal_points_per_letter * len(phrase.replace(" ", "")):
            result = total_points // average_points_per_letter
    if result == -1:
        # Assume the whole phrase
        result = len(phrase)
    elif result == 0:
        result = 1

    print("Assuming written text to be: {!r}".format(phrase[:result]))
    return result


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

    speed = []; pressure = []

    for word, stroke in strokes.items():
        for i, point in stroke.points.items():
            if i == 0:
                previous = point
                continue
            pressure.append(point.pressure)
            speed.append((point.co - previous.co).magnitude)
            previous = point

    number_written_letters = guess_written_phrase_size(strokes, speed)

    pressure_per_letter = average_value_per_letter(AUTOGRAPH_PHRASE, pressure, [0.3, 1.0], number_written_letters)
    speed_per_letter = average_value_per_letter(AUTOGRAPH_PHRASE, speed, [0.02, 0.08], number_written_letters)

    phrase_data = [{'pressure': p, 'speed': sp} for p, sp in zip(pressure_per_letter, speed_per_letter)]

    # print(f"\n\n\nSpeeds: {speed_per_letter}\n\npressures: {pressure_per_letter}")

    autograph_ignite(context, phrase_data, number_written_letters)


"""
# old autograph measuring code - would measure speed points after resampling curves -
# methods called here may be usefull to extract other parameters from the writting in the future


    #bpy.ops.gpencil.convert(type='POLY', use_timing_data=True)

    #anim_curve = bpy.data.curves[0].animation_data.action.fcurves[0]
    #points = bpy.data.curves[0].splines[0].points

    #anim_curve.convert_to_samples(0, 20000)
    #speed_per_stroke_curves = []
    #prev = None
    #for point in anim_curve.sampled_points:
        #if prev == None:
            #prev = point
            #continue
        #speed = (point.co - prev.co).y
        #speed_per_stroke_curves.append(speed)
        #prev = point

    #speed_per_stroke_curves = cleanup_speeds(speed_per_stroke_curves)

    # anum_curve = bpy.data.curves[0].animation_data.action.fcurves[0].sampled_points[i].co

"""

def autograph_ignite(context, phrase_data, number_written_letters):
    """Orchestrates the actual dance:

    the call to "assemble_actions" will pick the best action for each
    letter in the pre-defined phrase, based on the writting parameters
    measured and use Blenders capabilities to create a dinamic
    animation concatenating the actions.

    Then, it starts the dance!

    """
    total_frames = assemble_actions(context, AUTOGRAPH_PHRASE, phrase_data, number_written_letters)
    context.scene.frame_end = total_frames

    bpy.ops.screen.animation_play()



def concatenate_action(action, previous, ignore_height=True):
    """
    Changes target action curves by translating and rotating the
    root bone curves so that it continues smoothly from the point
    where the previous action had stopped.

    We do not do that for the "y" curve - the "height to the ground" for
    the root base by default, otherwise the character could
    "sink into the ground" after an action that ends in a lower plane

    """
    # print(f"concatenating actions {previous.name} and {action.name}")

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


def _get_int_or_default(dct, key, default=5):
    value = dct.get(key, "")
    return int(value) if value.isdigit() else default


def get_best_action(letter, letter_data):
    """Given features of choice, uses a vector space based
    on the mark for each of the features, and find the smallest
    distance of an action to the parameters of the current glyph
    on this space.

    Currently, features analyzed are hardcoded to pressure and speed

    """

    Features = namedtuple("Features", "pressure speed")

    pressure = letter_data.get("pressure", 0.5) * 10
    speed = letter_data.get("speed", 0.5) * 10

    feature_vector = Features(pressure, speed)

    plain_actions = ACTION_DATA[letter]
    indexed_actions = {}
    for action in plain_actions:
        pressure = _get_int_or_default(action, "pressure")
        speed = _get_int_or_default(action, "speed")
        indexed_actions[Features(pressure, speed)] = action

    def proximity(item):
        nonlocal feature_vector

        vector, _ = item
        distance = 0
        for component_1, component_2 in zip(vector, feature_vector):
            distance += (component_1 - component_2) ** 2
        distance **= 0.5
        return distance

    sorted_actions = sorted(indexed_actions.items(), key=proximity)

    # print(f"**** {letter} - {feature_vector} : {sorted_actions}")
    if sorted_actions:
        return sorted_actions[0][1]
    return None



def get_action_names(phrase, phrase_data):
    actions = []
    if not phrase_data:
        phrase_data = map(lambda x: {}, phrase)
    for letter, letter_data in zip(phrase, phrase_data):
        if letter not in ACTION_DATA:
            # If there is no action for a letter or punctuation on the target
            # phrase, just skip it.
            print("Could not find action for {letter!r} ".format(letter=letter))
            continue
        action = get_best_action(letter, letter_data)
        if not action:

            print("Could not match a good action for {letter!r} - picking random action".format(letter=letter))
            action = random.choice(ACTION_DATA[letter])
        actions.append(action)
    print(actions)
    return actions


def get_root_x_curve(action):
    """Find curve containing the X - coordinates for the Pelvis bone,
    to which all other coordinates are subordinated
    """
    for curve in action.fcurves:
        if curve.data_path == ROOT_X_NAME:
            break
    else:
        return None
    return curve


def get_final_x_location(action):
    """Get final x coordinate for the root bone in each action """
    curve = get_root_x_curve(action)
    if not curve:
        return 0
    points = curve.keyframe_points
    return points[-1].co


def assemble_actions(context, phrase, phrase_data=None, number_written_letters=len(AUTOGRAPH_PHRASE)):

    # Insert space actions at start and end of text to be danced:

    phrase = "{spaces}{phrase}{spaces}".format(phrase=phrase[:number_written_letters], spaces=" " * SPACE_MARGIN)
    if phrase_data:
        for i in range(SPACE_MARGIN):
            phrase_data.insert(0, phrase_data[0])
            phrase_data.append(phrase_data[-1])


    action_list = get_action_names(phrase, phrase_data)

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
    x_offset = 0
    total_actions = 0
    for action_data in action_list:
        action_name = action_data["name"]
        frames_str = action_data.get("frames")
        if frames_str.strip():
            frames = [int(v.strip()) for v in frames_str.split("-")]
        else:
            frames = None

        try:
            action = bpy.data.actions[action_name]
        except Exception as error:
            print("Expected action not found: ",  action_name)
            continue
        new_action = action.copy()

        new_action.name = "temp__{}__{}".format(action_data["letter"], action_name)

        def adjust_next_action(action, x_offset, frames):
            root_x_curve  = get_root_x_curve(action)
            for point in root_x_curve.keyframe_points:
                point.co[1] += x_offset
            return point.co[1]

        x_offset = adjust_next_action(new_action, x_offset, frames)
        try:
            strip = track.strips.new(action.name, previous_end, new_action)
        except RunTimeError as error:
            # print(error)
            print("Ignoring unknown runtime error for action {}. Skipping letter {}".format(action_name, action_data["letter"]))
            continue
        total_actions += 1
        if frames:
            strip.action_frame_start = frames[0]
            strip.action_frame_end = frames[1]
            total_frames = frames[1] - frames[0]
            # there might be flips
            flips = flipper.find_flips(new_action, frames[0], frames[1] + 1)
            if flips:
                flipper.invert_flips(new_action, flips, frames[0], frames[1] + 1)
        else:
            total_frames = new_action.frame_range[1]
        strip.select = False
        previous_end += total_frames + ACTION_SPACING
        prev_action = new_action


    previous_strip = None

    context.scene.objects.active = autograph

    if total_actions > 1:
        with switch_context_area(context, "NLA_EDITOR"), activate_layer(context, ARMATURE_LAYER):
            bpy.ops.nla.selected_objects_add()
            track.select = True
            bpy.ops.nla.select_all_toggle(True)
            bpy.ops.nla.transition_add()

    return previous_end - ACTION_SPACING



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
            return False
        print("Writting started")
        self.writting_started = True
        self.modal_func = self.check_writting_ended
        gp_layer.line_change = -1
        gp_layer.tint_color = POST_WRITTING_COLOR
        gp_layer.tint_factor = 1.0
        bpy.data.grease_pencil["GPencil"].layers["GP_Layer"].line_change = -1

        bpy.data.grease_pencil["GPencil"].palettes["GP_Palette"].colors["Color"].color = WRITTING_COLOR

        self.check_write_strokes = 0
        self.check_write_points = 0
        self.check_write_time = time.time()
        return False

    def check_writting_ended(self, context):
        """Unfortunately this could not be made to work -

        The idea would be to keep polling the grease-pencil objects, so that
        if a perceived time-lapse of ~3 seconds would pass with no new points
        added to the writting, signal the end of the written text
        and proceed to enact the dance.
        However, there is no way to pool a grease-pencil stroke _while_ it is been
        written - its length is always "0" even though writting is underway.

        the timeout is then increased so that at the writer is not interrupted mid-word.

        """
        # return False

        try:
            strokes = bpy.data.grease_pencil["GPencil"].layers["GP_Layer"].active_frame.strokes
        except (IndexError, KeyError, AttributeError):
            print("No writting - something went wrong")
            return True
        if len(strokes) and (len(strokes) > self.check_write_strokes or len(strokes[-1].points) > self.check_write_points):
            self.check_write_time = time.time()
            self.check_write_strokes = len(strokes)
            self.check_write_points = len(strokes[-1].points)

            return False

        if len(strokes) >= 1 and (time.time() - self.check_write_time) > STOPPED_WRITTING_TIMEOUT:
            if pyautogui:
                pyautogui.press("escape")
                autograph(context)
                # pyautogui.press("t")

            return True
        return False


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
