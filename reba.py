# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

import pdb; pdb.set_trace()
import argparse
import os
import json

import numpy as np
import quaternion
import pandas as pd

import src.kinect as kinect

[RIGHT_DIM, FRONT_DIM, UP_DIM] = range(3)

# angle unit [deg]
TRUNK_SCORE_TH_1 = 5
TRUNK_SCORE_TH_2 = 20
TRUNK_SCORE_TH_3 = 60
TRUNK_TWIST_TH = 20
TRUNK_SIDE_FLEX_TH = 5

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


def trans_pos(arr_pos):
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


def calc_proj(arr_v, e_x, e_y, e_z, dim):
    arr_v_proj =  np.stack(
        [
            np.sum(e_x * arr_v, axis=1),
            np.sum(e_y * arr_v, axis=1),
            np.sum(e_z * arr_v, axis=1)
        ],
        axis=1
    )                                       # (N, D)
    arr_v_proj[:, dim] = 0
    return arr_v_proj


def calc_deg(arr_v1, arr_v2):
    arr_cos_theta = np.sum(arr_v1 * arr_v2, axis=1)
    arr_cos_theta /= np.linalg(arr_v1, ord=2, axis=1)
    arr_cos_theta /= np.linalg(arr_v2, ord=2, axis=1)
    arr_rad = np.arccos(arr_cos_theta)
    arr_deg = np.degrees(arr_rad)

    return arr_deg


def calc_trunk_score(arr_pos, u_vert):
    # calculate basis
    hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]
    e_z = np.expand_dims(u_vert, axis=0)                # (1, D)
    e_y = np.cross(e_z, hip_l2r)                        # (N, D)
    e_y /= np.linalg.norm(hip_l2r, ord=2, axis=1, keepdims=True)
    e_x = np.cross(e_y, e_z)                            # (N, D)

    # calculate flexion or extension angles
    pel2nav = arr_pos[:, kinect.SPINE_NAVAL, :] - arr_pos[:, kinect.PELVIS, :]  # (N, D)
    pel2nav_proj_x = calc_proj(pel2nav, e_x, e_y, e_z, 0)
    flex_deg = calc_deg(pel2nav_proj_x, e_z)
    flex_deg[pel2nav_proj_x[:, 1] < 0] *= -1

    # calculate side flexion angles
    pel2nav_proj_y = calc_proj(pel2nav, e_x, e_y, e_z, 1)
    side_flex_deg = calc_deg(pel2nav_proj_y, e_z)

    # calculate twisting angles
    clav_l2r = arr_pos[:, kinect.CLAVICLE_RIGHT, :] - arr_pos[:, kinect.CLAVICLE_LEFT, :]
    twist_deg = calc_deg(clav_l2r, pel2nav)

    # calculate scores
    arr_score = np.ones(len(arr_pos))
    arr_score[np.abs(flex_deg) > TRUNK_SCORE_TH_1] += 1
    arr_score[np.abs(flex_deg) > TRUNK_SCORE_TH_2] += 1
    arr_score[flex_deg > TRUNK_SCORE_TH_3] += 1

    arr_score[
        (side_flex_deg > TRUNK_SIDE_FLEX_TH) |
        (twist_deg > TRUNK_TWIST_TH)
    ] += 1

    return arr_score


def calc_neck_score(arr_pos):
    # calculate basis
    clav_l2r = arr_pos[:, kinect.CLAVICLE_RIGHT, :] - arr_pos[:, kinect.CLAVICLE_LEFT, :]
    chest2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.SPINE_CHEST, :]
    e_z = chest2neck
    e_z /= np.linalg.norm(chest2neck, ord=2, axis=1, keepdims=True)     # (N, D)
    e_y = np.cross(e_z, clav_l2r)
    e_y /= np.linalg.norm(clav_l2r, ord=2, axis=1, keepdims=True)       # (N, D)
    e_x = np.cross(e_y, e_z)                                            # (N, D)

    # calculate flexion or extension angles
    neck2head = arr_pos[:, kinect.HEAD, :] - arr_pos[:, kinect.NECK, :]
    neck2head_proj_x = calc_proj(neck2head, e_x, e_y, e_z, 0)
    flex_deg = calc_deg(neck2head_proj_x, e_z)

    # calculate side flexion angles
    neck2head_proj_y = calc_proj(neck2head, e_x, e_y, e_z, 1)
    side_flex_deg = calc_deg(neck2head_proj_y, e_z)

    # calculate twisting angles
    ear_l2r = arr_pos[:, kinect.EAR_RIGHT, :] - arr_pos[:, kinect.EAR_LEFT, :]
    twist_deg = calc_deg(ear_l2r, clav_l2r)

    # calculate scores
    arr_score = np.ones(len(arr_pos))
    arr_score[flex_deg > NECK_SCORE_TH] += 1
    arr_score[
        (side_flex_deg > NECK_SIDE_FLEX_TH) |
        (twist_deg > NECK_TWIST_TH)
    ] += 1

    return arr_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=0)
    args = parser.parse_args()

    arr_time, arr_ori, arr_pos = kinect.read_time_ori_pos(args.input_json, args.body_id)

    arr_pos_trans = transform(arr_pos)

    pel2nav = arr_pos_trans[:, kinect.SPINE_NAVAL, :] - arr_pos_trans[:, kinect.PELVIS, :]
    hip_l2r = arr_pos_trans[:, kinect.HIP_RIGHT, :] - arr_pos_trans[:, kinect.HIP_LEFT, :]
    clav_l2r = arr_pos_trans[:, kinect.CLAVICLE_RIGHT, :] - arr_pos_trans[:, kinect.CLAVICLE_LEFT, :]

    trunk_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [FRONT_DIM, UP_DIM]])
    trunk_side_flex = calc_angle(np.array([[0, 1]]), pel2nav[:, [RIGHT_DIM, UP_DIM]])
    trunk_twist = calc_angle(hip_l2r, clav_l2r)

    


if __name__ == "__main__":
    main()
