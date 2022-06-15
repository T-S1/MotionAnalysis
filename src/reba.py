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
        self.WALKING_TH = 5            # [mm/s]
        self.SITTING_TH = 5

        self.UPPER_ARMS_SCORE_TH_1 = 20
        self.UPPER_ARMS_SCORE_TH_2 = 45
        self.UPPER_ARMS_SCORE_TH_3 = 90
        self.UPPER_ARMS_ABDUCT_TH = 20
        self.UPPER_ARMS_ROTATE_TH = 20
        self.SHOULDER_RAISE_TH = 5

        self.LOWER_ARMS_SCORE_TH_1 = 60
        self.LOWER_ARMS_SCORE_TH_2 = 100

        self.WRISTS_SCORE_TH = 15
        self.WRISTS_TWIST_TH = 45
        self.WRISTS_DEVIATE_TH = 5

        self.fps = fps

    def calc_unit_vec(self, arr_v):
        return arr_v / np.linalg.norm(arr_v, ord=2, axis=1, keepdims=True)

    def calc_vec_deg(self, arr_v1, arr_v2):
        arr_cos_theta = np.sum(arr_v1 * arr_v2, axis=1)
        arr_cos_theta /= np.linalg(arr_v1, ord=2, axis=1)
        arr_cos_theta /= np.linalg(arr_v2, ord=2, axis=1)
        arr_rad = np.arccos(arr_cos_theta)
        arr_deg = np.degrees(arr_rad)

        return arr_deg

    def estimate_vert(self, arr_pos):
        u_vert = np.zeros((1, 3))
        knee2hip_l = arr_pos[:, kinect.HIP_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2hip_r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        left_deg = self.calc_vec_deg(knee2hip_l, pel2neck)
        right_deg = self.calc_vec_deg(knee2hip_r, pel2neck)

        idx_min = np.argmin(left_deg + right_deg)
        u_vert[0, :] = pel2neck[idx_min, :]

        return u_vert

    def estimate_shoulder_base(self, arr_pos):
        chest2clav_l = arr_pos[:, kinect.CLAVICLE_LEFT, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        chest2clav_r = arr_pos[:, kinect.CLAVICLE_RIGHT, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        clav2shoul_l = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        clav2shoul_r = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]

        raised_deg_l = self.calc_vec_deg(-chest2clav_l, clav2shoul_l)
        raised_deg_r = self.calc_vec_deg(-chest2clav_r, clav2shoul_r)

        raised_deg_min = np.amin([raised_deg_l, raised_deg_r])

        return raised_deg_min

    def calc_basis(self, arr_vx, arr_vy_base):

        arr_vz = np.cross(arr_vx, arr_vy_base)
        arr_vy = np.cross(arr_vz, arr_vx)

        e_x = self.calc_unit_vec(arr_vx)
        e_y = self.calc_unit_vec(arr_vy)
        e_z = self.calc_unit_vec(arr_vz)

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

    def calc_trunk_score(self, arr_pos, u_vert):
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.PELVIS, :]

        flex_deg = self.calc_vec_deg(pel2neck, u_vert)

        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        arr_basis = self.calc_basis(u_vert, hip_l2r)
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

    def calc_legs_score(self, arr_pos, u_vert):
        # calculate symmetry
        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]

        arr_basis = self.calc_basis(u_vert, hip_l2r)
        u_norm = arr_basis[:, :, 1]

        hip2knee_l = arr_pos[:, kinect.KNEE_LEFT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        hip2knee_r = arr_pos[:, kinect.KNEE_RIGHT, :] - arr_pos[:, kinect.HIP_RIGHT, :]

        hip2knee_l_mirror = np.sum(hip2knee_l * u_norm, axis=1, keepdims=True)
        hip2knee_l_mirror *= u_norm
        hip2knee_deg = self.calc_vec_deg(hip2knee_r, hip2knee_l_mirror)

        knee2ank_l = arr_pos[:, kinect.ANKLE_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2ank_r = arr_pos[:, kinect.ANKLE_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        knee2ank_l_mirror = np.sum(knee2ank_l * u_norm, axis=1, keepdims=True)
        knee2ank_l_mirror *= u_norm
        knee2ank_deg = self.calc_vec_deg(knee2ank_r, knee2ank_l_mirror)

        mean_deg = (hip2knee_deg + knee2ank_deg) / 2

        # check walking
        dif_ank_l = arr_pos[1:, kinect.ANKLE_LEFT, :] - arr_pos[:-1, kinect.ANKLE_LEFT, :]
        dif_ank_r = arr_pos[1:, kinect.ANKLE_RIGHT, :] - arr_pos[:-1, kinect.ANKLE_RIGHT, :]

        speed_ank_l = dif_ank_l * self.fps
        speed_ank_r = dif_ank_r * self.fps

        walking = (speed_ank_l > self.WALKING_TH) | (speed_ank_r > self.WALKING_TH)

        # check sitting
        hdeg_hip2knee_l = self.calc_vec_deg(hip2knee_l, u_vert) - 90
        sitting = (hdeg_hip2knee_l < self.SITTING_TH)

        knee_l_deg = self.calc_vec_deg(hip2knee_l, knee2ank_l)
        knee_r_deg = self.calc_vec_deg(hip2knee_r, knee2ank_r)

        # calcluate score
        arr_score = np.ones(len(arr_pos))
        arr_score[(mean_deg > self.LEGS_BILATERAL_TH) & (~walking) & (~sitting)] = 2
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_1) |
            (knee_r_deg > self.KNEE_ANGLE_TH_1)
        ] += 1
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_2) |
            (knee_r_deg > self.KNEE_ANGLE_TH_2)
        ] += 1

        return arr_score

    def calc_arms_rot_deg(self, arr_pos):
        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.ELBOW_RIGHT, :]
        clav2shoul_l = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.CLAVICLE_LEFT, :]
        clav2shoul_r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.CLAVICLE_RIGHT, :]

        u_norm_l = -np.cross(clav2shoul_l, shoul2elb_l)
        u_norm_r = np.cross(clav2shoul_r, shoul2elb_r)

        # avoid zero vector
        u_norm_l[np.all(u_norm_l == 0, axis=1)] = np.array([0, 1, 0])
        u_norm_r[np.all(u_norm_r == 0, axis=1)] = np.array([0, 1, 0])

        u_norm_l = self.calc_unit_vec(u_norm_l)
        u_norm_r = self.calc_unit_vec(u_norm_r)

        u_shoul2elb_l = self.calc_unit_vec(shoul2elb_l)
        u_shoul2elb_r = self.calc_unit_vec(shoul2elb_r)
        u_elb2wrist_l = self.calc_unit_vec(elb2wrist_l)
        u_elb2wrist_r = self.calc_unit_vec(elb2wrist_r)
        elb2wrist_l_proj = u_elb2wrist_l - np.sum(u_elb2wrist_l * u_shoul2elb_l, axis=1) * u_shoul2elb_l
        elb2wrist_r_proj = u_elb2wrist_r - np.sum(u_elb2wrist_r * u_shoul2elb_r, axis=1) * u_shoul2elb_r

        # avoid zero vector
        elb2wrist_l_proj[np.all(u_norm_l == 0, axis=1)] = np.array([0, 1, 0])
        elb2wrist_r_proj[np.all(u_norm_l == 0, axis=1)] = np.array([0, 1, 0])

        u_proj_l = self.calc_unit_vec(elb2wrist_l_proj)
        u_proj_r = self.calc_unit_vec(elb2wrist_r_proj)

        rot_deg_l = self.calc_vec_deg(u_proj_l, u_norm_l)
        rot_deg_r = self.calc_vec_deg(u_proj_r, u_norm_r)

        return rot_deg_l, rot_deg_r

    def calc_upper_arms_score(self, arr_pos, shoul_base, supported=None):
        if supported == None:
            supported = np.array([False] * len(arr_pos))

        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        pel2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.PELVIS, :]
        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        clav2shoul_l = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        clav2shoul_r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]

        arr_basis = self.calc_basis(pel2neck, shoul_l2r)
        shoul2elbow_l_rot = self.rot_coord(shoul2elb_l, arr_basis)
        shoul2elbow_r_rot = self.rot_coord(shoul2elb_r, arr_basis)

        # front flexion is positive
        flex_rad_l = np.arctan2(shoul2elbow_l_rot[:, 2], -shoul2elbow_l_rot[:, 0])
        flex_rad_r = np.arctan2(shoul2elbow_r_rot[:, 2], -shoul2elbow_r_rot[:, 0])
        flex_deg_l = np.degrees(flex_rad_l)
        flex_deg_r = np.degrees(flex_rad_r)

        flex_deg = flex_deg_l
        idx_r = (np.abs(flex_deg_l) < np.abs(flex_deg_r))
        flex_deg[idx_r] = flex_deg_r[idx_r]

        # abduction is positive
        abduct_rad_l = np.arctan2(shoul2elbow_l_rot[:, 1], -shoul2elbow_l_rot[:, 0])
        abduct_rad_r = np.arctan2(-shoul2elbow_r_rot[:, 1], -shoul2elbow_l_rot[:, 0])
        abduct_deg_l = np.degrees(abduct_rad_l)
        abduct_deg_r = np.degrees(abduct_rad_r)

        rot_deg_l, rot_deg_r = self.calc_arms_rot_deg(arr_pos)

        chest2clav_l = arr_pos[:, kinect.CLAVICLE_LEFT, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        chest2clav_r = arr_pos[:, kinect.CLAVICLE_RIGHT, :] - arr_pos[:, kinect.SPINE_CHEST, :]

        shoul_deg_l = self.calc_vec_deg(-chest2clav_l, clav2shoul_l)
        shoul_deg_r = self.calc_vec_deg(-chest2clav_r, clav2shoul_r)

        raised_deg_l = shoul_deg_l - shoul_base
        raised_deg_r = shoul_deg_r - shoul_base

        arr_score_l = np.ones(len(arr_pos))
        arr_score_l[np.abs(flex_deg_l) > self.UPPER_ARMS_SCORE_TH_1] = 2
        arr_score_l[flex_deg_l > self.UPPER_ARMS_SCORE_TH_2] = 3
        arr_score_l[flex_deg_l > self.UPPER_ARMS_SCORE_TH_3] = 4
        arr_score_l[
            (abduct_deg_l > self.UPPER_ARMS_ABDUCT_TH) |
            (rot_deg_l > self.UPPER_ARMS_ROTATE_TH)
        ] += 1
        arr_score_l[raised_deg_l > self.SHOULDER_RAISE_TH] += 1

        arr_score_r = np.ones(len(arr_pos))
        arr_score_r[np.abs(flex_deg_r) > self.UPPER_ARMS_SCORE_TH_1] = 2
        arr_score_r[flex_deg_r > self.UPPER_ARMS_SCORE_TH_2] = 3
        arr_score_r[flex_deg_r > self.UPPER_ARMS_SCORE_TH_3] = 4
        arr_score_r[
            (abduct_deg_r > self.UPPER_ARMS_ABDUCT_TH) |
            (rot_deg_r > self.UPPER_ARMS_ROTATE_TH)
        ] += 1
        arr_score_r[raised_deg_r > self.SHOULDER_RAISE_TH] += 1
        arr_score_r[supported] += 1

        return arr_score_l, arr_score_r

    def calc_lower_arms_score(self, arr_pos):
        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]

        flex_deg_l = self.calc_vec_deg(shoul2elb_l, elb2wrist_l)
        flex_deg_r = self.calc_vec_deg(shoul2elb_r, elb2wrist_r)

        arr_score_l = np.ones(len(arr_pos))
        arr_score_l[
            (flex_deg_l < self.LOWER_ARMS_SCORE_TH_1) |
            (flex_deg_l > self.LOWER_ARMS_SCORE_TH_2)
        ] = 2

        arr_score_r = np.ones(len(arr_pos))
        arr_score_r[
            (flex_deg_r < self.LOWER_ARMS_SCORE_TH_1) |
            (flex_deg_r > self.LOWER_ARMS_SCORE_TH_2)
        ] = 2

        return arr_score_l, arr_score_r

    def calc_wrists_twist(self, arr_pos):
        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.ELBOW_RIGHT, :]
        wrist2thumb_l = arr_pos[:, kinect.THUMB_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2thumb_r = arr_pos[:, kinect.THUMB_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]

        v_norm_l = np.cross(shoul2elb_l, elb2wrist_l)
        v_norm_r = np.cross(shoul2elb_r, elb2wrist_r)

        # avoid zero vector
        v_norm_l[np.all(v_norm_l == 0, axis=1)] = np.array([0, 1, 0])
        v_norm_r[np.all(v_norm_r == 0, axis=1)] = np.array([0, 1, 0])

        u_elb2wrist_l = self.calc_unit_vec(v_norm_l)
        u_elb2wrist_r = self.calc_unit_vec(v_norm_r)
        u_wrist2thumb_l = self.calc_unit_vec(wrist2thumb_l)
        u_wrist2thumb_r = self.calc_unit_vec(wrist2thumb_r)
        wrist2thumb_l_proj = u_wrist2thumb_l - np.sum(u_wrist2thumb_l * u_elb2wrist_l, axis=1) * u_elb2wrist_l
        wrist2thumb_r_proj = u_wrist2thumb_r - np.sum(u_wrist2thumb_r * u_elb2wrist_r, axis=1) * u_elb2wrist_r

        twist_deg_l = self.calc_vec_deg(wrist2thumb_l_proj, v_norm_l)
        twist_deg_l = self.calc_vec_deg(wrist2thumb_r_proj, v_norm_r)
        # ?

        ##### from here


    def calc_wrists_score(self, arr_pos):
        wrist2hand_l = arr_pos[:, kinect.HAND_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2hand_r = arr_pos[:, kinect.HAND_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]
        wrist2thumb_l = arr_pos[:, kinect.THUMB_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2thumb_r = arr_pos[:, kinect.THUMB_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[: kinect.ELBOW_RIGHT, :]

        flex_deg_l = self.calc_vec_deg(wrist2hand_l, elb2wrist_l)
        flex_deg_r = self.calc_vec_deg(wrist2hand_r, elb2wrist_r)

        arr_basis_l = self.calc_basis(elb2wrist_l, wrist2thumb_l)
        arr_basis_r = self.calc_basis(elb2wrist_r, wrist2thumb_r)
        wrist2hand_l_rot = self.rot_coord(wrist2hand_l, arr_basis_l)
        wrist2hand_r_rot = self.rot_coord(wrist2hand_r, arr_basis_r)

        deviate_rad_l = np.arctan2(wrist2hand_l_rot[:, 1], wrist2hand_l_rot[:, 0])
        deviate_rad_r = np.arctan2(wrist2hand_r_rot[:, 1], wrist2hand_r_rot[:, 0])
        deviate_deg_l = np.degrees(deviate_rad_l)
        deviate_deg_r = np.degrees(deviate_rad_r)

        arr_score_l = np.ones(len(arr_pos))
        arr_score_l[flex_deg_l > self.WRISTS_SCORE_TH] = 2
        arr_score_l[
            (flex_deg_l > self.WRISTS_TWIST_TH) |
            (deviate_deg_l > self.WRISTS_DEVIATE_TH)
        ] += 1

        arr_score_r = np.ones(len(arr_pos))
        arr_score_r[flex_deg_r > self.WRISTS_SCORE_TH] = 2
        arr_score_r[
            (flex_deg_r > self.WRISTS_TWIST_TH) |
            (deviate_deg_r > self.WRISTS_DEVIATE_TH)
        ] += 1

        return arr_basis_l, arr_basis_r
        ##### from here


# code sample
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=0)
    args = parser.parse_args()
