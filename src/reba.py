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


class Reba:
    def __init__(self, fps=30):

        [self.RIGHT_DIM, self.FRONT_DIM, self.UP_DIM] = range(3)

        # angle unit [deg]
        self.TRUNK_SCORE_TH_1 = 5
        self.TRUNK_SCORE_TH_2 = 20
        self.TRUNK_SCORE_TH_3 = 60
        self.TRUNK_TWIST_TH = 20
        self.TRUNK_SIDE_FLEX_TH = 5

        self.NECK_SCORE_TH = 20
        self.NECK_TWIST_TH = 20
        self.NECK_SIDE_FLEX_TH = 20

        self.KNEE_ANGLE_TH_1 = 30
        self.KNEE_ANGLE_TH_2 = 60
        self.LEGS_BILATERAL_TH = 20
        self.WALKING_TH = 5             # [mm/s]
        self.SITTING_THIGH_TH = 80

        self.UPPER_ARMS_SCORE_TH_1 = 20
        self.UPPER_ARMS_SCORE_TH_2 = 45
        self.UPPER_ARMS_SCORE_TH_3 = 90
        self.UPPER_ARMS_ABDUCT_TH = 20
        self.UPPER_ARMS_ROTATE_TH = 20
        self.SHOULDER_RAISE_TH = 5

        self.LOWER_ARMS_SCORE_TH_1 = 60
        self.LOWER_ARMS_SCORE_TH_2 = 100

        self.WRISTS_SCORE_TH = 15

        self.fps = fps
        self.u_vert = np.array([[0, 0, 1]])

    def calc_vec_deg(self, arr_v1, arr_v2):
        arr_cos_theta = np.sum(arr_v1 * arr_v2, axis=1)
        arr_cos_theta /= np.linalg(arr_v1, ord=2, axis=1)
        arr_cos_theta /= np.linalg(arr_v2, ord=2, axis=1)
        arr_rad = np.arccos(arr_cos_theta)
        arr_deg = np.degrees(arr_rad)

        return arr_deg

    def estimate_vert(self, arr_pos):
        knee2hip_l = arr_pos[:, kinect.HIP_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2hip_r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        left_deg = self.calc_vec_deg(knee2hip_l, pel2neck)
        right_deg = self.calc_vec_deg(knee2hip_r, pel2neck)

        idx_min = np.argmin(left_deg + right_deg)
        self.u_vert[0, :] = pel2neck[idx_min, :]

    def calc_basis(self, arr_vx, arr_vy_base):

        arr_vz = np.cross(arr_vx, arr_vy_base)
        arr_vy = np.cross(arr_vz, arr_vx)

        e_x = arr_vx / np.linalg.norm(arr_vx, ord=2, axis=1, keepdims=True)
        e_y = arr_vy / np.linalg.norm(arr_vy, ord=2, axis=1, keepdims=True)
        e_z = arr_vz / np.linalg.norm(arr_vz, ord=2, axis=1, keepdims=True)

        return np.stack([e_x, e_y, e_z], axis=2)

    def rot_coord(self, arr_vt, arr_basis):
        e_x = arr_basis[:, :, 0]
        e_y = arr_basis[:, :, 1]
        e_z = arr_basis[:, :, 2]
        rot_mat = np.stack(
            [
                np.sum(e_x * arr_vt, axis=1),
                np.sum(e_y * arr_vt, axis=1),
                np.sum(e_z * arr_vt, axis=1)
            ],
            axis=2
        )                                               # (N, D, xyz)
        arr_vt_temp = np.expand_dims(arr_vt, axis=1)    # (N, 1, D)
        arr_vt_rot = np.matmul(arr_vt_temp, rot_mat)
        return arr_vt_rot

    def calc_trunk_score(self, arr_pos):
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.PELVIS, :]

        flex_deg = self.calc_vec_deg(pel2neck, self.u_vert)

        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        arr_basis = self.calc_basis(self.u_vert, hip_l2r)
        pel2neck_rot = self.rot_coord(pel2neck, arr_basis)

        flex_rad = np.arctan2(pel2neck_rot[:, 2], pel2neck[:, 0])
        flex_deg = np.degrees(flex_rad)

        side_flex_rad = np.arctan2(pel2neck_rot[:, 1], pel2neck_rot[:, 0])
        side_flex_deg = np.degrees(side_flex_rad)

        arr_basis = self.calc_basis(hip_l2r, pel2neck)
        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul_l2r_rot = self.rot_coord(shoul_l2r, arr_basis)
        twist_rad = np.arctan2(shoul_l2r_rot[:, 2], shoul_l2r[:, 0])
        twist_deg = np.degrees(twist_rad)

        arr_score = np.ones(len(arr_pos))
        arr_score[np.abs(flex_deg) > self.TRUNK_SCORE_TH_1] = 2
        arr_score[np.abs(flex_deg) > self.TRUNK_SCORE_TH_2] = 3
        arr_score[flex_deg > self.TRUNK_SCORE_TH_3] = 4

        arr_score[
            (np.abs(side_flex_deg) > self.TRUNK_SIDE_FLEX_TH) |
            (np.abs(twist_deg) > self.TRUNK_TWIST_TH)
        ] += 1

        return arr_score

    def calc_neck_score(self, arr_pos):
        neck2head = arr_pos[:, kinect.HEAD, :] - arr_pos[:, kinect.NECK, :]
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.PELVIS, :]
        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]

        arr_basis = self.calc_basis(pel2neck, shoul_l2r)
        neck2head_rot = self.rot_coord(neck2head, arr_basis)
        flex_rad = np.arctan2(neck2head_rot[:, 2], neck2head_rot[:, 0])
        flex_deg = np.degrees(flex_rad)

        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        arr_basis = self.calc_basis(pel2neck, shoul_l2r)
        neck2head_rot = self.rot_coord(neck2head, arr_basis)
        side_flex_rad = np.arctan2(neck2head_rot[:, 1], neck2head_rot[:, 0])
        side_flex_deg = np.degrees(side_flex_rad)

        arr_basis = self.calc_basis(shoul_l2r, pel2neck)
        neck2head_rot = self.rot_coord(neck2head, arr_basis)
        twist_rad = np.arctan2(neck2head_rot[:, 2], neck2head_rot[:, 0])
        twist_deg = np.degrees(twist_rad)

        arr_score = np.ones(len(arr_pos))
        arr_score[(flex_deg > self.NECK_SCORE_TH) | (flex_deg < 0)] = 2

        arr_score[
            (np.abs(side_flex_deg) > self.NECK_SIDE_FLEX_TH) |
            (np.abs(twist_deg) > self.NECK_TWIST_TH)
        ] += 1

        return arr_score

    def calc_legs_score(self, arr_pos):
        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]

        arr_basis = self.calc_basis(self.u_vert, hip_l2r)
        u_norm = arr_basis[:, :, 1]

        hip2knee_l = arr_pos[:, kinect.KNEE_LEFT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        hip2knee_r = arr_pos[:, kinect.KNEE_RIGHT, :] - arr_pos[:, kinect.HIP_RIGHT, :]

        pel = arr_pos[:, kinect.PELVIS, :]

        hip2knee_l_mirror = np.sum(hip2knee_l * u_norm, axis=1, keepdims=True)
        hip2knee_l_mirror *= u_norm
        hip2knee_deg = self.calc_vec_deg(hip2knee_r, hip2knee_l_mirror)

        knee2ank_l = arr_pos[:, kinect.ANKLE_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2ank_r = arr_pos[:, kinect.ANKLE_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        knee2ank_l_mirror = np.sum(knee2ank_l * u_norm, axis=1, keepdims=True)
        knee2ank_l_mirror *= u_norm
        knee2ank_deg = self.calc_vec_deg(knee2ank_r, knee2ank_l_mirror)

        mean_deg = (hip2knee_deg + knee2ank_deg) / 2

        dif_ank_l = arr_pos[1:, kinect.ANKLE_LEFT, :] - arr_pos[:-1, kinect.ANKLE_LEFT, :]
        dif_ank_r = arr_pos[1:, kinect.ANKLE_RIGHT, :] - arr_pos[:-1, kinect.ANKLE_RIGHT, :]

        speed_ank_l = dif_ank_l * self.fps
        speed_ank_r = dif_ank_r * self.fps

        walking = (speed_ank_l > self.WALKING_TH) | (speed_ank_r > self.WALKING_TH)

        knee_l_deg = self.calc_vec_deg(hip2knee_l, knee2ank_l)
        knee_r_deg = self.calc_vec_deg(hip2knee_r, knee2ank_r)

        arr_score = np.ones(len(arr_pos))
        arr_score[(mean_deg > self.LEGS_BILATERAL_TH) & (~walking)] = 2
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_1) |
            (knee_r_deg > self.KNEE_ANGLE_TH_1)
        ] += 1
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_2) |
            (knee_r_deg > self.KNEE_ANGLE_TH_2)
        ] += 1
        ##### from here


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
