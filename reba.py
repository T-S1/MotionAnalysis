# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

import pdb; pdb.set_trace()
import argparse
import os
import json

import numpy as np
import pandas as pd

import src.kinect as kinect

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


def estimate_vert(arr_pos):
    pel = arr_pos[:, kinect.PELVIS, :]            # (T, 3)
    nav = arr_pos[:, kinect.SPINE_NAVAL, :]
    hip_l = arr_pos[:, kinect.HIP_LEFT, :]
    hip_r = arr_pos[:, kinect.HIP_RIGHT, :]
    knee_l = arr_pos[:, kinect.KNEE_LEFT, :]
    knee_r = arr_pos[:, kinect.KNEE_RIGHT, :]

    d_pel2nav = np.linalg.norm(nav - pel, ord=2, axis=1, keepdims=True)             # (T, 1)
    d_knee2hip_l = np.linalg.norm(hip_l - knee_l, ord=2, axis=1, keepdims=True)
    d_knee2hip_r = np.linalg.norm(hip_r - knee_r, ord=2, axis=1, keepdims=True)

    u_pel2nav = (nav - pel) / d_pel2nav                 # (T, 3)
    u_knee2hip_l = (hip_l - knee_l) / d_knee2hip_l
    u_knee2hip_r = (hip_r - knee_r) / d_knee2hip_r

    i_vert = np.argmax(
        np.sum(u_pel2nav * (u_knee2hip_l + u_knee2hip_r), axis=1)
    )

    return u_pel2nav[i_vert]    # (3, )


def transform(arr_pos):
    u_vert = estimate_vert(arr_pos)             # (3, )

    hip_l = arr_pos[:, kinect.HIP_LEFT, :]
    hip_r = arr_pos[:, kinect.HIP_RIGHT, :]

    hip_l2r = hip_r - hip_l                                     # (T, 3)
    e_u = np.tile(u_vert, (len(arr_pos), 1))                    # (T, 3)
    e_f = np.cross(e_u, hip_l2r)                                # (T, 3)
    e_f /= np.linalg.norm(e_f, ord=2, axis=1, keepdims=True)    # (T, 3)
    e_r = np.cross(e_f, e_u)                                    # (T, 3)

    basis = np.zeros((len(arr_pos), 3, 3))      # (T, 3, 3)
    basis[:, :, RIGHT_DIM] = e_r
    basis[:, :, FRONT_DIM] = e_f
    basis[:, :, UP_DIM] = e_u

    arr_pos_trans = np.matmul(arr_pos, basis)     # (T, J, 3)

    return arr_pos_trans


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
