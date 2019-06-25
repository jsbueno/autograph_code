import os, sys, re
import random
import threading, time
import statistics

from collections import namedtuple
from contextlib import contextmanager
from itertools import islice

import bpy
from bpy.props import StringProperty, PointerProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup

import flipper
import parameter_reader


AUTOGRAPH_PHRASE = "escrever com o corpo"
AUTOGRAPH_ORIGINAL_PHRASE = AUTOGRAPH_PHRASE

INTENSITIES_TABLE_URL = "https://docs.google.com/spreadsheets/d/1R-GADr8HBUqiawQVrBgW_0_h-9bJMXO1kdI7qs9It3g/export?format=csv"

SPACE_MARGIN_START = 0
SPACE_MARGIN_END = 3

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
FROZEN_ACTION_DATA = ACTION_DATA

WRITTING_FADE_FRAME = 90

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
    "version": (1, 0, 0)
}


def scene_cleanup(context):
    """AKA: Nuclear Blast

    removes all grease pencils, temporary actions, NLA tracks
    and resets the camera position - everything ready
    to a fresh text to be drawn.
    """

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
    # Reset camera
    # Camera reseting is important because grease-pencil coordinates
    # used for size and speed calculations depend on absolute coordinates
    bpy.data.objects["Camera"].location = (2, -4, 1)
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            break


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

def normalize_f(values, normalize):
    factor =  1 / (normalize[1] - normalize[0])
    return [(v - normalize[0]) * factor for v in values]

def average_value_per_letter(
    phrase, measured_points, normalize=(0, 1),
    number_written_letters=len(AUTOGRAPH_PHRASE)
):
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
    text_length = len(trimmed_phrase)
    factor = len(measured_points) / (text_length or 1)
    if factor < 1:
        return measured_points
    new_points = []
    i = 0
    for letter in phrase:
        if letter == " ":
            # for spaces, copy parameters from the previous drawn letter
            new_points.append(new_points[-1] if new_points else 0)
            continue
        points_value = sum(measured_points[int(i * factor): int((i + 1) * factor)]) / factor
        new_points.append(points_value)
        i += 1
    return normalize_f(new_points, normalize)


def guess_written_phrase_size(strokes, speed):

    phrase = AUTOGRAPH_PHRASE
    phrase = phrase.replace(" ", "")

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

    total_spaces = 0
    original_glyphs = iter(AUTOGRAPH_PHRASE)

    for letter, glyph in zip(phrase, original_glyphs):
        while glyph == " ":
            total_spaces += 1
            glyph = next(original_glyphs, ".")
    result += total_spaces
    print("Assuming written text to be: {!r}".format(AUTOGRAPH_PHRASE[:result]))
    return result

def _format_list(lst):
    return "[{}]".format(", ".join("{:0.3f}".format(el) for el in lst ))


def autograph(context):
    """
    Main functionality -
    this will extract writting metrics from the active grease-pencil
    layer, and use those to select the proper actions to be
    associated with the main Autograph actor.
    """
    global ACTION_DATA

    try:
        strokes = bpy.data.grease_pencil[0].layers[0].active_frame.strokes
    except IndexError:
        print("Can't start: No grease pencil writting found on scene.")
        return

    speed = []; pressure = []; size = []; psize = []
    average_points_per_letter = 42

    size_counter = 0
    tmp_size_max, tmp_size_min = -1000, 1000

    for word, stroke in strokes.items():
        for i, point in stroke.points.items():
            if i == 0:
                previous = point
                continue
            pressure.append(point.pressure)
            speed.append((point.co - previous.co).magnitude)
            previous = point
            z = point.co[2]
            if z > tmp_size_max:
                tmp_size_max = z
            if z < tmp_size_min:
                tmp_size_min = z

            size_counter += 1
            if size_counter > average_points_per_letter:
                size.extend([tmp_size_max - tmp_size_min] * size_counter)
                psize.append(tmp_size_max - tmp_size_min)
                tmp_size_max, tmp_size_min = -1000, 1000
                size_counter = 0

    print("MEASURED SIZES", _format_list(psize))

    if size_counter:
        size.extend([tmp_size_max - tmp_size_min] * size_counter)

    # TODO: take in account letter size for "number_written_letters" -
    # it is widelly innacurate for small text.
    number_written_letters = guess_written_phrase_size(strokes, speed)

    pressure_per_letter = average_value_per_letter(AUTOGRAPH_PHRASE, pressure, [0.3, 1.0], number_written_letters)
    raw_speed_per_letter = average_value_per_letter(
        AUTOGRAPH_PHRASE, speed, [0.1, 0.4],
        # [context.scene.autograph_text.lower_speed, context.scene.autograph_text.upper_speed],
        number_written_letters)

    size_per_letter =  average_value_per_letter(AUTOGRAPH_PHRASE, size, [0.05, 0.5], number_written_letters)

    writting_time = context.scene.autograph_text.total_writting_time
    letter_per_time = number_written_letters / writting_time
    speed_factor = letter_per_time / statistics.median(raw_speed_per_letter)

    speed_per_letter = normalize_f([s * speed_factor for s in raw_speed_per_letter ], (0.7, 2.7))

    phrase_data = [{'pressure': p, 'speed': sp, 'size': size} for p, sp, size in zip(pressure_per_letter, speed_per_letter,  size_per_letter)]

    print("pressure data:  \n size: {sizes}\n, pressures: {pressures}\n, speed: {speeds}\n".format(
            pressures=_format_list(pressure_per_letter),
            sizes = _format_list(size_per_letter),
            speeds = _format_list(speed_per_letter)
        )
    )
    print("raw_speeds:", _format_list(raw_speed_per_letter))
    print("Total writting time: {:0.2f}, num. letters: {}".format(
            writting_time,
            number_written_letters
        )
    )

    try:
        print("Downloading ACTION DATA from {}".format(INTENSITIES_TABLE_URL))
        ACTION_DATA = parameter_reader.get_online_actions(INTENSITIES_TABLE_URL)
    except RuntimeError as error:
        print(error)
        print("Failed refreshing online ACTION_DATA")
    else:
        print("Ok")

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
    context.scene.frame_start = 1
    context.scene.frame_end = total_frames
    fade_text()
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


def get_best_action(letter, letter_data, isolate_actions):
    """Given features of choice, uses a vector space based
    on the mark for each of the features, and find the smallest
    distance of an action to the parameters of the current glyph
    on this space.

    Currently, features analyzed are hardcoded to pressure and speed

    """

    Features = namedtuple("Features", "pressure speed size")

    pressure = letter_data.get("pressure", 0.5) * 10
    speed = letter_data.get("speed", 0.5) * 10
    size = letter_data.get("size", 0.5) * 10

    feature_vector = Features(pressure, speed, size)

    plain_actions = ACTION_DATA[letter]
    indexed_actions = {}
    for action in plain_actions:
        if isolate_actions and not action.get("selected", "").strip():
            continue
        pressure = _get_int_or_default(action, "pressure")
        speed = _get_int_or_default(action, "speed")
        size = _get_int_or_default(action, "size")
        indexed_actions[Features(pressure, speed, size)] = action

    def proximity(item):
        nonlocal feature_vector

        vector, _ = item
        distance = 0
        for component_1, component_2 in zip(vector, feature_vector):
            distance += (component_1 - component_2) ** 2
        distance **= 0.5
        return distance

    sorted_actions = sorted(indexed_actions.items(), key=proximity)

    if sorted_actions:
        # print("**" * 50, "\n", letter, feature_vector, sorted_actions, "\n", "##" * 50)
        return sorted_actions[0][1]
    return None



def get_action_names(phrase, phrase_data):
    actions = []
    isolate_actions = bpy.context.scene.autograph_text.isolate_actions
    if not phrase_data:
        phrase_data = map(lambda x: {}, phrase)
    for letter, letter_data in zip(phrase, phrase_data):
        if letter not in ACTION_DATA:
            # If there is no action for a letter or punctuation on the target
            # phrase, just skip it.
            print("Could not find action for {letter!r} ".format(letter=letter))
            continue
        action = get_best_action(letter, letter_data, isolate_actions)

        if not action:
            if not isolate_actions:
                print("Could not match a good action for {letter!r} - picking random action".format(letter=letter))
                action = random.choice(ACTION_DATA[letter])
            else:
                print("Letter '{}' not selected in spreadsheet - skipped".format(letter))
                continue
        actions.append((action, letter_data))
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


def _normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r"[^a-zA-Z0-9#_]", "_", name)
    while "__" in name:
        name = name.replace("__", "_")
    return name

def _normalized_list(keys):
    result = {}
    for key in keys:
        result.setdefault(_normalize_name(key), []).append(key)
    return result

_NORMALIZED_ACTIONS = None

def get_action(name):
    global _NORMALIZED_ACTIONS
    if not _NORMALIZED_ACTIONS:
        _NORMALIZED_ACTIONS = _normalized_list(bpy.data.actions.keys())
    if name in bpy.data.actions:
        return bpy.data.actions[name]
    n_name = _normalize_name(name)
    if n_name in _NORMALIZED_ACTIONS:
        res = _NORMALIZED_ACTIONS[n_name]
        if len(res) > 1:
            print("Using ambiguous action name {}. Options are {}".format(name, res))
        return bpy.data.actions[res[0]]
    raise KeyError(name)


def adjust_next_action(action, x_offset, frames, reverse_movement):
    """
    Adds x-offset to action copy composing dance.
    Returns offset of final frame in action

    reverse_movement means the frames within the action will be
    used in reverse order by the NLA
    """
    root_x_curve  = get_root_x_curve(action)
    if frames:
        if frames[0] <= 1:
            frames[0] = 1
        if frames[1] >= len(root_x_curve.keyframe_points):
            frames[1] = len(root_x_curve.keyframe_points) - 1
    if reverse_movement:
        x_offset -= root_x_curve.keyframe_points[frames[1]].co[1]
    elif frames:
        x_offset -= root_x_curve.keyframe_points[frames[0]].co[1]

    for point in root_x_curve.keyframe_points:
        point.co[1] += x_offset

    if not frames:
        return point.co[1]

    last_x_position = frames[1] if not reverse_movement else frames[0]
    return root_x_curve.keyframe_points[last_x_position].co[1]

    # Alternative attempt: modify only points in curve that are to be used:

    #if not frames:
        #frames = None, None
    #else:
        #frames = list(frames)
        #frames[1] += 1

    #for point in islice(root_x_curve.keyframe_points, *frames):
        #point.co[1] += x_offset

    #if frames[0] is not None:
        #root_x_curve.keyframe_points[0].co[1] = root_x_curve.keyframe_points[frames[0]].co[1]
        #root_x_curve.keyframe_points[-1].co[1] = root_x_curve.keyframe_points[frames[1] - 1].co[1]

def convert_action_to_samples(action):
    for curve in action.fcurves:
        if len(curve.keyframe_points) > 1 and not len(curve.sampled_points):
            curve.convert_to_samples(0, curve.keyframe_points[-1].co[0])


def assemble_actions(context, phrase, phrase_data=None, number_written_letters=len(AUTOGRAPH_PHRASE)):

    autograph = bpy.data.objects[AUTOGRAPH_ID]
    # autograph.hide = False

    # Insert space actions at start and end of text to be danced:
    phrase = "{spaces_start}{phrase}{spaces_end}".format(
        phrase=phrase[:number_written_letters],
        spaces_start=" " * SPACE_MARGIN_START,
        spaces_end=" " * SPACE_MARGIN_END
    )
    if phrase_data:
        for i in range(SPACE_MARGIN_START):
            phrase_data.insert(0, phrase_data[0])
        for i in range(SPACE_MARGIN_END):
            phrase_data.append(phrase_data[-1])

    action_list = get_action_names(phrase, phrase_data)

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
    for (action_data, letter_data) in action_list:
        action_name = action_data["name"]
        frames_str = action_data.get("frames")
        reverse_movement = False
        if frames_str.strip():
            frames = [int(v.strip()) for v in frames_str.split("-")]
            frame_start, frame_end = frames
            if frame_start > frame_end:
                reverse_movement = True
                frame_start, frame_end = frame_end, frame_start
        else:
            frames = None

        try:
            action = get_action(action_name)
        except KeyError as error:
            print("Expected action not found: ",  action_name)
            continue
        new_action = action.copy()

        new_action.name = "temp__{}__{}".format(action_data["letter"], action_name)
        print("{}: , x_offset: {}".format(new_action.name, x_offset))
        x_offset = adjust_next_action(new_action, x_offset, frames, reverse_movement)

        convert_action_to_samples(new_action)

        try:
            strip = track.strips.new(new_action.name, previous_end, new_action)
        except RuntimeError as error:
            # print(error)
            print("Ignoring unknown runtime error for action {}. Skipping letter {}\n\n{}".format(action_name, action_data["letter"], error))
            continue
        total_actions += 1
        if frames:
            if reverse_movement:
                strip.use_reverse = True

            strip.action_frame_start = frame_start
            strip.action_frame_end = frame_end
            total_frames = frame_end - frame_start
            # there might be flips
            flips = flipper.find_flips(new_action, frame_start, frame_end + 1)
            if flips:
                flipper.invert_flips(new_action, flips, frame_start, frame_end + 1)
        else:
            total_frames = new_action.frame_range[1]
        try:
            speed_factor = float(action_data.get("speed_factor", "1").strip())
        except ValueError:
            speed_factor = 1.0

        action_spacing = ACTION_SPACING
        if letter_data["speed"] > 0.75:
            superspeed = (letter_data["speed"] - 0.75) * 4
            speed_factor *= 1 / (1 + superspeed)
            action_spacing = int(ACTION_SPACING * 0.6 + 4 * (1 - superspeed))
        elif letter_data["speed"] < 0.25:
            action_spacing = int(ACTION_SPACING + 40 * (0.25 - max(0, letter_data["speed"])))
        if speed_factor != 1.0:
            print("{!r} using speed_factor {}".format(action_data["letter"], speed_factor))
            strip.scale = speed_factor

        total_frames = strip.frame_end - strip.frame_start

        strip.select = False
        previous_end += total_frames + action_spacing
        prev_action = new_action

    previous_strip = None

    context.scene.objects.active = autograph

    _add_transitions(context, track, total_actions)
    camera_setup(x_offset)
    # autograph.hide = True

    return previous_end - ACTION_SPACING


def camera_setup(max_x):
    print("final x_offset", max_x)
    camera = bpy.data.objects["Camera"]
    camera.location[0] = -max_x / 2
    camera.location[1] = min(-4, max_x * 1.1)
    camera.location[2] = 1


def _add_transitions(context, track, total_actions):
        if total_actions <= 1:
            return
        with switch_context_area(context, "NLA_EDITOR"):
            bpy.ops.nla.selected_objects_add()
            track.select = True
            bpy.ops.nla.select_all_toggle(True)
            bpy.ops.nla.transition_add()


def autograph_test(context):
    with activate_layer(context, ARMATURE_LAYER):
        total_frames = assemble_actions(context, AUTOGRAPH_PHRASE)
    context.scene.frame_end = total_frames

    bpy.ops.screen.animation_play()


def fade_text():
    grease = bpy.data.grease_pencil["GPencil"]
    grease.animation_data_create()
    act = bpy.data.actions.new("GPencil.001Action")
    grease.animation_data.action = act
    curve = act.fcurves.new(data_path="""palettes["GP_Palette"].colors["Color"].alpha""")

    curve.keyframe_points.insert(1, 1)
    curve.keyframe_points.insert(WRITTING_FADE_FRAME, 0)


class RepeatAutograph(Operator):
    """Dança Novamente"""

    bl_idname = "autograph.repeat"
    bl_label = "Reproduzir último"
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


class Autograph(Operator):
    """Inicia Autograph"""

    bl_idname = "autograph.start"
    bl_label = "AUTOGRAPH"
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
        grease = bpy.data.grease_pencil["GPencil"]
        grease.layers["GP_Layer"].line_change = -1
        grease.layers["GP_Layer"].parent = bpy.data.objects["Camera"]

        grease.palettes["GP_Palette"].colors["Color"].color = WRITTING_COLOR

        self.check_write_strokes = 0
        self.check_write_points = 0
        self.start_writting_time = self.check_write_time = time.time()
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
            total_time = time.time() - self.start_writting_time - STOPPED_WRITTING_TIMEOUT
            context.scene.autograph_text.total_writting_time = total_time
            if pyautogui:
                pyautogui.press("escape")
                autograph(context)


            return True
        return False


    def modal(self, context, event):
        if event.type == "TIMER":
            if self.modal_func(context):
                return {"FINISHED"}

        return {"PASS_THROUGH"}



class AutographText(bpy.types.PropertyGroup):

    def update_text_parameter(self, context):
        global AUTOGRAPH_PHRASE
        AUTOGRAPH_PHRASE = self.text
        print("Phrase changed to '{}'".format(self.text))

    text = bpy.props.StringProperty(
        default=AUTOGRAPH_ORIGINAL_PHRASE,
        name="text",
        description="Texto que será dançado",
        update=update_text_parameter
    )

    isolate_actions = bpy.props.BoolProperty(
        default=False,
        name="isolate_actions",
        description="Usar somente ações selecionadas",
    )

    lower_speed = bpy.props.FloatProperty(
        default=0.004,
        name="lower_speed",
        description="valor inferior da limitacao da velocidade",
    )
    upper_speed = bpy.props.FloatProperty(
        default=0.017,
        name="uper_speed",
        description="valor inferior da limitacao da velocidade",
    )
    total_writting_time = bpy.props.FloatProperty(
        name="total_writting_time",
        description="duração total da última escrita",
    )



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

        layout.label(text="")
        row = layout.row()
        row.operator("autograph.start")
        row.scale_y = 3.0
        row = layout.row()
        row.operator("autograph.repeat")
        row = layout.row()
        row.prop(scene.autograph_text, "text", text="Texto")
        row = layout.row()
        row.prop(scene.autograph_text, "isolate_actions", text="Isolar ações")
        #row = layout.row()
        #row.prop(scene.autograph_text, "lower_speed", text="vel. baixo")
        #row = layout.row()
        #row.prop(scene.autograph_text, "upper_speed", text="vel. alto")



def register():
    print("Registering Autograph add-on")
    bpy.utils.register_module(__name__)
    bpy.types.Scene.autograph_text = bpy.props.PointerProperty(type=AutographText)


def unregister():
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
