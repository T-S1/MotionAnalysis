# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

# import pdb; pdb.set_trace()
import argparse
import os

import numpy as np
import pandas as pd

import src.reader as reader

OUTPUT_DIR = "./csv"

JOINT_LIST = [
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=1)
    args = parser.parse_args()

    arr_timestamp, arr_orientations, arr_positions = reader.read_time_ori_pos(
        args.input_json, args.body_id
    )

    orientation_columns = ["timestamp"] + [
        f"{joint_name}_{wxyz}"
        for joint_name in JOINT_LIST
        for wxyz in ["w", "x", "y", "z"]
    ]
    position_columns = ["timestamp"] + [
        f"{joint_name}_{xyz}"
        for joint_name in JOINT_LIST
        for xyz in ["x", "y", "z"]
    ]
    dir_name = os.path.splitext(os.path.basename(args.input_json))[0]

    if not os.path.isdir(f"{OUTPUT_DIR}/{dir_name}"):
        os.makedirs(f"{OUTPUT_DIR}/{dir_name}")

    arr_orientations = np.reshape(arr_orientations, (-1, len(JOINT_LIST) * 4))  # (T, J * 4)
    arr_positions = np.reshape(arr_positions, (-1, len(JOINT_LIST) * 3))        # (T, J * 3)

    arr_timestamp = np.expand_dims(arr_timestamp, axis=1)     # (T, 1)

    orientations_table = np.concatenate([arr_timestamp, arr_orientations], axis=1)
    positions_table = np.concatenate([arr_timestamp, arr_positions], axis=1)

    df_orientations = pd.DataFrame(orientations_table, columns=orientation_columns)
    df_positions = pd.DataFrame(positions_table, columns=position_columns)

    if not os.path.isdir(f"{OUTPUT_DIR}/{dir_name}/body_{args.body_id}"):
        os.makedirs(f"{OUTPUT_DIR}/{dir_name}/body_{args.body_id}")

    df_orientations.to_csv(
        f"{OUTPUT_DIR}/{dir_name}/body_{args.body_id}/orientations.csv",
        index=False
    )
    df_positions.to_csv(
        f"{OUTPUT_DIR}/{dir_name}/body_{args.body_id}/positions.csv",
        index=False
    )


if __name__ == "__main__":
    main()
