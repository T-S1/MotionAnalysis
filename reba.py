# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

import pdb; pdb.set_trace()
import argparse
import os
import json

import numpy as np
import pandas as pd

import src.reader as reader

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

[RIGHT_DIM, FRONT_DIM, UP_DIM] = range(3)

# angle unit [deg]
TRUNK_SCORE_TH_1 = 20
TRUNK_SCORE_TH_2 = 60
TRUNK_TWIST_TH = 20
TRUNK_SIDE_FLEX_TH = 20

NECK_SCORE_TH = 20
NECK_TWIST_TH = 20
NECK_SIDE_FLEX_TH = 20

LEGS_ANGLE_TH_1 = 30
LEGS_ANGLE_TH_2 = 60
LEGS_UNIRATERAL_TH = 20
SITTING_THIGH_TH = 80

UPPER_ARMS_SCORE_TH_1 = 20
UPPER_ARMS_SCORE_TH_2 = 45
UPPER_ARMS_SCORE_TH_3 = 90
UPPER_ARMS_ABDUCT_TH = 20
UPPER_ARMS_ROTATE_TH = 20
SHOULDER_RAISE_TH = 5

LOWER_ARMS_SCORE_TH_1 = 60
LOWER_ARMS_SCORE_TH_2 = 100

WRISTS_SCORE_TH = 15
WRISTS_TWIST_



def read_time_pos(input_json, target_body_id):

    with open(input_json, "r") as f:
        track_map = json.load(f)

    timestamp_list = []
    positions_list = []

    frames = track_map["frames"]
    for i in range(len(frames)):
        frame = frames[i]
        bodies = frame["bodies"]
        timestamp_usec = frame["timestamp_usec"]
        num_bodies = frame["num_bodies"]
        for j in range(num_bodies):
            body = bodies[j]
            body_id = body["body_id"]

            if body_id == target_body_id:

                joint_positions = body["joint_positions"]

                timestamp_list.append(timestamp_usec)
                positions_list.append(joint_positions)

    num_frames = len(timestamp_list)

    arr_timestamp = np.zeros(num_frames)
    arr_positions = np.zeros([num_frames, NUM_JOINTS, 3])

    arr_timestamp[:] = np.array(timestamp_list)
    arr_positions[:, :, :] = np.array(positions_list)         # (T, J, 3)

    return arr_timestamp, arr_positions


def estimate_vert(arr_positions):
    pel = arr_positions[:, PELVIS, :]
    nav = arr_positions[:, SPINE_NAVAL, :]
    hip_l = arr_positions[:, HIP_LEFT, :]
    hip_r = arr_positions[:, HIP_RIGHT, :]
    knee_l = arr_positions[:, KNEE_LEFT, :]
    knee_r = arr_positions[:, KNEE_RIGHT, :]

    u_pel2nav = (nav - pel) / np.linalg.norm(nav - pel, ord=2, axis=1)
    u_knee2hip_l = (hip_l - knee_l) / np.linalg.norm(hip_l - knee_l, ord=2, axis=1)
    u_knee2hip_r = (hip_r - knee_r) / np.linalg.norm(hip_r - knee_r, ord=2, axis=1)

    i_vert = np.argmax(
        np.dot(u_pel2nav, u_knee2hip_l) + np.dot(u_pel2nav, u_knee2hip_r)
    )

    return u_pel2nav[i_vert]    # (3, 3)


def transform(arr_positions):
    u_vert = estimate_vert(arr_positions)

    arr_basis = np.zeros((len(arr_positions), 3, 3))

    arr_hip_l2r = arr_positions[:, HIP_RIGHT, :] - arr_positions[:, HIP_LEFT, :]
    arr_e_u = np.tile(u_vert, (len(arr_positions), 1))
    arr_e_f = np.cross(arr_e_u, arr_hip_l2r)
    arr_e_f /= np.linalg.norm(arr_e_f, ord=2, axis=1)
    arr_e_r = np.cross(arr_e_f, arr_e_u)

    arr_basis = np.zeros((len(arr_positions), 3, 3))
    arr_basis[:, RIGHT_DIM, :] = arr_e_r
    arr_basis[:, FRONT_DIM, :] = arr_e_f
    arr_basis[:, UP_DIM, :] = arr_e_u

    arr_basis_inv = np.linalg.inv(arr_basis)                    # (T, 3, 3)
    arr_basis_inv_t = np.expand_dims(arr_basis_inv, axis=0)     # (1, T, 3, 3)

    arr_positions_t = np.transpose(arr_positions, (1, 0, 2))    # (J, T, 3)
    arr_positions_t = np.expand_dims(arr_positions_t, axis=2)   # (J, T, 1, 3)

    arr_k = np.matmul(arr_positions, arr_basis_inv)     # (J, T, 1, 3)
    arr_k = np.squeeze(arr_k)                           # (J, T, 3)
    arr_k = np.transpose(arr_k, (1, 0, 2))              # (T, J, 3)

    return arr_k


def calc_proj_angle(arr_positions, joint, dim):

    pel2nav = arr_positions[:, SPINE_NAVAL, :] - arr_positions[:, PELVIS, :]

    proj_r = pel2nav[:, [FRONT_DIM, UP_DIM]]
    u_proj_r = proj_r / np.linalg.norm(proj_r, ord=2, axis=1)
    flex_angle = np.arccos(u_proj_r[:, 1])

    proj_f = pel2nav[:, [RIGHT_DIM, UP_DIM]]
    u_proj_f = proj_f / np.linalg.norm(proj_f, ord=2, axis=1)
    side_angle = np.arccos(u_proj_f[:, 1])

    return flex_angle, side_angle


def calc_angle(arr_v1, arr_v2):
    arr_norm1 = np.linalg.norm(arr_v1, ord=2, axis=1)
    arr_norm2 = np.linalg.norm(arr_v2, ord=2, axis=1)
    arr_cos_theta = np.dot(arr_v1, arr_v2) / (arr_norm1 * arr_norm2)
    arr_theta = np.arccos(arr_cos_theta)

    return arr_theta    # (T,)


def calc_trunc_score(arr_pos_trans):
    


def reba(arr_positions, load_force, coupling):

    arr_pos_trans = transform(arr_positions)

    pel2nav = arr_pos_trans[:, SPINE_NAVAL, :] - arr_pos_trans[:, PELVIS, :]
    hip_l2r = arr_pos_trans[:, HIP_RIGHT, :] - arr_pos_trans[:, HIP_LEFT, :]
    clav_l2r = arr_pos_trans[:, CLAVICLE_RIGHT, :] - arr_pos_trans[:, CLAVICLE_LEFT, :]

    trunk_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [FRONT_DIM, UP_DIM]])
    trunk_side_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [RIGHT_DIM, UP_DIM]])
    trunk_twist = calc_angle(hip_l2r, clav_l2r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=0)
    args = parser.parse_args()

    arr_timestamp, arr_positions = read_time_pos(args.input_json, args.body_id)

    arr_pos_trans = transform(arr_positions)

    pel2nav = arr_pos_trans[:, SPINE_NAVAL, :] - arr_pos_trans[:, PELVIS, :]
    hip_l2r = arr_pos_trans[:, HIP_RIGHT, :] - arr_pos_trans[:, HIP_LEFT, :]
    clav_l2r = arr_pos_trans[:, CLAVICLE_RIGHT, :] - arr_pos_trans[:, CLAVICLE_LEFT, :]

    trunk_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [FRONT_DIM, UP_DIM]])
    trunk_side_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [RIGHT_DIM, UP_DIM]])
    trunk_twist = calc_angle(hip_l2r, clav_l2r)

    


if __name__ == "__main__":
    main()
