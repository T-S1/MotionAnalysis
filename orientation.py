# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

import pdb; pdb.set_trace()
import argparse
import os
import json

import numpy as np
import pandas as pd

OUTPUT_DIR = "./csv"

NUM_JOINTS = 32

JOINT_LIST = [
    PELVIS,
    SPINE_NAVAL,
    SPINE_CHEST,
    NECK,
    CLAVICLE_LEFT,
    SHOULDER_LEFT,
    ELBOW_LEFT,
    WRIST_LEFT,
    HAND_LEFT,
    HANDTIP_LEFT,
    THUMB_LEFT,
    CLAVICLE_RIGHT,
    SHOULDER_RIGHT,
    ELBOW_RIGHT,
    WRIST_RIGHT,
    HAND_RIGHT,
    HANDTIP_RIGHT,
    THUMB_RIGHT,
    HIP_LEFT,
    KNEE_LEFT,
    ANKLE_LEFT,
    FOOT_LEFT,
    HIP_RIGHT,
    KNEE_RIGHT,
    ANKLE_RIGHT,
    FOOT_RIGHT,
    HEAD,
    NOSE,
    EYE_LEFT,
    EAR_LEFT,
    EYE_RIGHT,
    EAR_RIGHT,
] = range(NUM_JOINTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        track_map = json.load(f)

    timestamp_usec_map = {}
    joint_orientations_map = {}
    joint_positions_map = {}

    frames = track_map["frames"]
    for i in range(len(frames)):
        frame = frames[i]
        bodies = frame["bodies"]
        timestamp_usec = frame["timestamp_usec"]
        num_bodies = frame["num_bodies"]
        for j in range(num_bodies):
            body = bodies[j]
            body_id = body["body_id"]
            joint_orientations = body["joint_orientations"]
            joint_positions = body["joint_positions"]

            if body_id not in joint_orientations_map:
                timestamp_usec_map[body_id] = []
                joint_orientations_map[body_id] = []
                joint_positions_map[body_id] = []

            timestamp_usec_map[body_id].append(timestamp_usec)
            joint_orientations_map[body_id].append(joint_orientations)
            joint_positions_map[body_id].append(joint_positions)

    arr_timestamp_map = {}
    arr_orientations_map = {}
    arr_positions_map = {}
    for body_id in joint_orientations_map:

        num_frames = len(timestamp_usec_map[body_id])

        arr_timestamp = np.zeros(num_frames)
        arr_orientations = np.zeros([num_frames, NUM_JOINTS, 4])
        arr_positions = np.zeros([num_frames, NUM_JOINTS, 3])

        arr_timestamp[:] = np.array(timestamp_usec_map[body_id])
        arr_orientations[:, :, :] = np.array(joint_orientations_map[body_id])   # (T, J, 4)
        arr_positions[:, :, :] = np.array(joint_positions_map[body_id])         # (T, J, 3)

        arr_timestamp_map[body_id] = arr_timestamp
        arr_orientations_map[body_id] = arr_orientations
        arr_positions_map[body_id] = arr_positions

    


if __name__ == "__main__":
    main()
