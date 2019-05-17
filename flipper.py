import bpy
from itertools import islice


def sgn(f):
    return -1 if f < 0 else 1


def get_quaternion_curves(action):
    quaternion_curves = []
    for curve in action.fcurves:
        if "pelvis" in curve.data_path and "quaternion" in curve.data_path:
            quaternion_curves.append(curve)
    if len(quaternion_curves) != 4:
        print("erro - curvas não encontradas")
        return []
    return quaternion_curves


def find_flips(action, start=None, stop=None):

    y_quat = get_quaternion_curves(action)[2]
    previous_co = None
    flips = []
    for point in islice(y_quat.keyframe_points, start, stop):
        if previous_co == None:
            previous_co = point.co[1]
            if previous_co < 0:
                flips.append(1)
            continue

        if sgn(point.co[1]) != sgn(previous_co) and abs(point.co[1] - previous_co) > 0.3:
            flips.append(point.co[0])    

        previous_co = point.co[1]
    return flips


def invert_flips(action, flips, start=None, stop=None):
    quat = get_quaternion_curves(action)
    inverting = False
    flips = iter(flips)
    next_flip = next(flips, 10000000)
    for quat_points in zip(*[islice(curve.keyframe_points, start, stop) for curve in quat]):
        frame = quat_points[0].co[0]
        if frame == next_flip:
            inverting = not inverting
            next_flip = next(flips, 100000000)
        if inverting:
            for point in quat_points:
                point.co[1] *= -1
                point.handle_left[1] *= -1
                point.handle_right[1] *= -1


if __name__ == "__main__":
    for action in bpy.data.actions:
        flips = find_flips(action)
        if not flips:
            continue
        print("invertendo :", action.name)

        invert_flips(action, flips)
        
