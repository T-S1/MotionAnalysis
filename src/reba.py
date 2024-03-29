# Referances
# [1] definition of orientation
#     https://microsoft.github.io/Azure-Kinect-Body-Tracking/release/1.1.x/unionk4a__quaternion__t.html

# import pdb; pdb.set_trace()
import argparse
import os

import numpy as np
import pandas as pd

import src.kinect as kinect


class Reba:
    def __init__(self, fps=30):

        [self.RIGHT_DIM, self.FRONT_DIM, self.UP_DIM] = range(3)

        # angle unit [deg]
        self.TRUNK_SCORE_TH_1 = 5
        self.TRUNK_SCORE_TH_2 = 20
        self.TRUNK_SCORE_TH_3 = 60
        self.TRUNK_TWIST_TH = 15
        self.TRUNK_SIDE_FLEX_TH = 10

        self.NECK_SCORE_TH = 20
        self.NECK_TWIST_TH = 20
        self.NECK_SIDE_FLEX_TH = 5

        self.KNEE_ANGLE_TH_1 = 30
        self.KNEE_ANGLE_TH_2 = 60
        self.LEGS_BILATERAL_TH = 15
        self.WALKING_TH = 100            # [mm/s]
        self.SITTING_TH = 10

        self.UPPER_ARMS_SCORE_TH_1 = 20
        self.UPPER_ARMS_SCORE_TH_2 = 45
        self.UPPER_ARMS_SCORE_TH_3 = 90
        self.UPPER_ARMS_ABDUCT_TH = 15
        self.UPPER_ARMS_ROTATE_TH = 60
        self.SHOULDER_RAISE_TH = 20

        self.LOWER_ARMS_SCORE_TH_1 = 60
        self.LOWER_ARMS_SCORE_TH_2 = 100

        self.WRISTS_SCORE_TH = 15
        self.WRISTS_TWIST_TH = 90
        self.WRISTS_DEVIATE_TH = 30

        self.table_a = np.array([
            [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
            [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
            [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
            [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
            [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
        ])  # Trunk, Neck, Legs

        self.table_b = np.array([
            [[1, 2, 2], [1, 2, 3]],
            [[1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 5]],
            [[4, 5, 5], [5, 6, 7]],
            [[6, 7, 8], [7, 8, 8]],
            [[7, 8, 8], [8, 9, 9]]
        ])  # Upper arm, Lower arm, Wrist

        self.table_c = np.array([
            [ 1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],
            [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],
            [ 2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],
            [ 3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],
            [ 4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9,  9],
            [ 6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],
            [ 7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],
            [ 8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],
            [ 9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
        ])  # A, B

        self.fps = fps

        self.info_map = {}

    def calc_unit_vec(self, arr_v):
        return arr_v / np.linalg.norm(arr_v, ord=2, axis=1, keepdims=True)

    def calc_vec_deg(self, arr_v1, arr_v2):
        arr_u1 = self.calc_unit_vec(arr_v1)
        arr_u2 = self.calc_unit_vec(arr_v2)
        arr_cos_theta = np.sum(arr_u1 * arr_u2, axis=1)
        arr_rad = np.arccos(arr_cos_theta)
        arr_deg = np.degrees(arr_rad)

        return arr_deg

    def estimate_vert(self, arr_pos):
        u_vert = np.zeros((1, 3))
        knee2hip_l = arr_pos[:, kinect.HIP_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2hip_r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]
        pel2nav = arr_pos[:, kinect.SPINE_NAVAL, :] - arr_pos[:, kinect.PELVIS, :]

        left_deg = self.calc_vec_deg(knee2hip_l, pel2nav)
        right_deg = self.calc_vec_deg(knee2hip_r, pel2nav)

        idx_min = np.argmin(left_deg + right_deg)
        u_vert[0, :] = pel2nav[idx_min, :]

        return u_vert

    def estimate_shoulder_base(self, arr_pos):
        chest2clav_l = arr_pos[:, kinect.CLAVICLE_LEFT, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        chest2clav_r = arr_pos[:, kinect.CLAVICLE_RIGHT, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        clav2shoul_l = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.CLAVICLE_LEFT, :]
        clav2shoul_r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.CLAVICLE_RIGHT, :]

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

        return [e_x, e_y, e_z]

    def rot_coord(self, arr_vt, arr_basis):
        e_x = arr_basis[0]
        e_y = arr_basis[1]
        e_z = arr_basis[2]
        arr_vt_rot = np.stack(
            [
                np.sum(e_x * arr_vt, axis=1),
                np.sum(e_y * arr_vt, axis=1),
                np.sum(e_z * arr_vt, axis=1)
            ],
            axis=1
        )                                               # (N, D)
        return arr_vt_rot

    def calc_trunk_score(self, arr_pos, u_vert):
        pel2nav = arr_pos[:, kinect.SPINE_NAVAL, :] - arr_pos[:, kinect.PELVIS, :]
        knee2hip_l = arr_pos[:, kinect.HIP_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2hip_r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        flex_deg = np.maximum(
            self.calc_vec_deg(knee2hip_l, pel2nav),
            self.calc_vec_deg(knee2hip_r, pel2nav)
        )

        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        arr_basis = self.calc_basis(u_vert, hip_l2r)
        pel2nav_rot = self.rot_coord(pel2nav, arr_basis)

        side_flex_rad = np.arctan2(pel2nav_rot[:, 1], pel2nav_rot[:, 0])
        side_flex_deg = np.degrees(side_flex_rad)

        arr_basis = self.calc_basis(hip_l2r, pel2nav)
        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul_l2r_rot = self.rot_coord(shoul_l2r, arr_basis)
        twist_rad = np.arctan2(shoul_l2r_rot[:, 2], -shoul_l2r[:, 0])
        twist_deg = np.degrees(twist_rad)

        arr_score = np.ones(len(arr_pos), dtype=int)
        arr_score[np.abs(flex_deg) > self.TRUNK_SCORE_TH_1] = 2
        arr_score[np.abs(flex_deg) > self.TRUNK_SCORE_TH_2] = 3
        arr_score[flex_deg > self.TRUNK_SCORE_TH_3] = 4

        arr_score[
            (np.abs(side_flex_deg) > self.TRUNK_SIDE_FLEX_TH) |
            (np.abs(twist_deg) > self.TRUNK_TWIST_TH)
        ] += 1

        self.info_map["trunk_flex_deg"] = flex_deg
        self.info_map ["trunk_side_flex_deg"] = side_flex_deg
        self.info_map["trunk_twist_deg"] = twist_deg

        return arr_score

    def calc_neck_score(self, arr_pos):
        neck2head = arr_pos[:, kinect.HEAD, :] - arr_pos[:, kinect.NECK, :]
        chest2neck = arr_pos[:, kinect.NECK, :] - arr_pos[:, kinect.SPINE_CHEST, :]
        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        head2nose = arr_pos[:, kinect.NOSE, :] - arr_pos[:, kinect.HEAD, :]

        arr_basis = self.calc_basis(chest2neck, shoul_l2r)
        neck2head_rot = self.rot_coord(neck2head, arr_basis)
        flex_rad = np.arctan2(neck2head_rot[:, 2], neck2head_rot[:, 0])
        flex_deg = np.degrees(flex_rad)

        shoul_l2r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        arr_basis = self.calc_basis(chest2neck, shoul_l2r)
        neck2head_rot = self.rot_coord(neck2head, arr_basis)
        side_flex_rad = np.arctan2(neck2head_rot[:, 1], neck2head_rot[:, 0])
        side_flex_deg = np.degrees(side_flex_rad)

        arr_basis = self.calc_basis(chest2neck, shoul_l2r)
        head2nose_rot = self.rot_coord(head2nose, arr_basis)
        twist_rad = np.arctan2(head2nose_rot[:, 1], head2nose_rot[:, 2])
        twist_deg = np.degrees(twist_rad)

        arr_score = np.ones(len(arr_pos), dtype=int)
        arr_score[(flex_deg > self.NECK_SCORE_TH) | (flex_deg < 0)] = 2

        arr_score[
            (np.abs(side_flex_deg) > self.NECK_SIDE_FLEX_TH) |
            (np.abs(twist_deg) > self.NECK_TWIST_TH)
        ] += 1

        self.info_map["neck_flex_deg"] = flex_deg
        self.info_map["neck_side_flex_deg"] = side_flex_deg
        self.info_map["neck_twist_deg"] = twist_deg

        return arr_score

    def calc_legs_score(self, arr_pos, u_vert):
        # calculate symmetry
        hip_l2r = arr_pos[:, kinect.HIP_RIGHT, :] - arr_pos[:, kinect.HIP_LEFT, :]

        arr_basis = self.calc_basis(u_vert, hip_l2r)
        u_norm = arr_basis[1]

        hip2knee_l = arr_pos[:, kinect.KNEE_LEFT, :] - arr_pos[:, kinect.HIP_LEFT, :]
        hip2knee_r = arr_pos[:, kinect.KNEE_RIGHT, :] - arr_pos[:, kinect.HIP_RIGHT, :]

        hip2knee_l_mirror = np.sum(hip2knee_l * u_norm, axis=1, keepdims=True) * u_norm
        hip2knee_deg = self.calc_vec_deg(hip2knee_r, hip2knee_l_mirror)

        knee2ank_l = arr_pos[:, kinect.ANKLE_LEFT, :] - arr_pos[:, kinect.KNEE_LEFT, :]
        knee2ank_r = arr_pos[:, kinect.ANKLE_RIGHT, :] - arr_pos[:, kinect.KNEE_RIGHT, :]

        knee2ank_l_mirror = np.sum(knee2ank_l * u_norm, axis=1, keepdims=True) * u_norm
        knee2ank_deg = self.calc_vec_deg(knee2ank_r, knee2ank_l_mirror)

        mean_deg = (hip2knee_deg + knee2ank_deg) / 2

        legs_deg = self.calc_vec_deg(hip2knee_l, hip2knee_r)

        # check walking
        dif_ank_l = np.zeros((len(arr_pos), 3))
        dif_ank_r = np.zeros((len(arr_pos), 3))
        dif_ank_l[1:] = arr_pos[1:, kinect.ANKLE_LEFT, :] - arr_pos[:-1, kinect.ANKLE_LEFT, :]
        dif_ank_r[1:] = arr_pos[1:, kinect.ANKLE_RIGHT, :] - arr_pos[:-1, kinect.ANKLE_RIGHT, :]

        speed_ank_l = np.linalg.norm(dif_ank_l * self.fps, ord=2, axis=1)
        speed_ank_r = np.linalg.norm(dif_ank_l * self.fps, ord=2, axis=1)

        walking = (speed_ank_l > self.WALKING_TH) | (speed_ank_r > self.WALKING_TH)

        # check sitting
        hdeg_hip2knee_l = self.calc_vec_deg(hip2knee_l, u_vert) - 90
        sitting = (hdeg_hip2knee_l < self.SITTING_TH)

        knee_l_deg = self.calc_vec_deg(hip2knee_l, knee2ank_l)
        knee_r_deg = self.calc_vec_deg(hip2knee_r, knee2ank_r)

        # calcluate score
        arr_score = np.ones(len(arr_pos), dtype=int)
        arr_score[(legs_deg > self.LEGS_BILATERAL_TH) & (~walking) & (~sitting)] = 2
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_1) |
            (knee_r_deg > self.KNEE_ANGLE_TH_1)
        ] += 1
        arr_score[
            (knee_l_deg > self.KNEE_ANGLE_TH_2) |
            (knee_r_deg > self.KNEE_ANGLE_TH_2)
        ] += 1

        self.info_map["legs_deg"] = legs_deg
        self.info_map["speed_ank_l"] = speed_ank_l
        self.info_map["speed_ank_r"] = speed_ank_r
        self.info_map["sitting_deg"] = hdeg_hip2knee_l
        self.info_map["knee_l_deg"] = knee_l_deg
        self.info_map["knee_r_deg"] = knee_r_deg

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
        arr_inner_l = np.sum(u_elb2wrist_l * u_shoul2elb_l, axis=1, keepdims=True)
        arr_inner_r = np.sum(u_elb2wrist_r * u_shoul2elb_r, axis=1, keepdims=True)
        elb2wrist_l_proj = u_elb2wrist_l - arr_inner_l * u_shoul2elb_l
        elb2wrist_r_proj = u_elb2wrist_r - arr_inner_r * u_shoul2elb_r

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
        clav2shoul_l = arr_pos[:, kinect.SHOULDER_LEFT, :] - arr_pos[:, kinect.CLAVICLE_LEFT, :]
        clav2shoul_r = arr_pos[:, kinect.SHOULDER_RIGHT, :] - arr_pos[:, kinect.CLAVICLE_RIGHT, :]

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

        arr_score_l = np.ones(len(arr_pos), dtype=int)
        arr_score_l[np.abs(flex_deg_l) > self.UPPER_ARMS_SCORE_TH_1] = 2
        arr_score_l[flex_deg_l > self.UPPER_ARMS_SCORE_TH_2] = 3
        arr_score_l[flex_deg_l > self.UPPER_ARMS_SCORE_TH_3] = 4
        arr_score_l[
            (abduct_deg_l > self.UPPER_ARMS_ABDUCT_TH) |
            (rot_deg_l > self.UPPER_ARMS_ROTATE_TH)
        ] += 1
        arr_score_l[raised_deg_l > self.SHOULDER_RAISE_TH] += 1

        arr_score_r = np.ones(len(arr_pos), dtype=int)
        arr_score_r[np.abs(flex_deg_r) > self.UPPER_ARMS_SCORE_TH_1] = 2
        arr_score_r[flex_deg_r > self.UPPER_ARMS_SCORE_TH_2] = 3
        arr_score_r[flex_deg_r > self.UPPER_ARMS_SCORE_TH_3] = 4
        arr_score_r[
            (abduct_deg_r > self.UPPER_ARMS_ABDUCT_TH) |
            (rot_deg_r > self.UPPER_ARMS_ROTATE_TH)
        ] += 1
        arr_score_r[raised_deg_r > self.SHOULDER_RAISE_TH] += 1
        arr_score_r[supported] += 1

        self.info_map["uarm_flex_deg_l"] = flex_deg_l
        self.info_map["uarm_abduct_deg_l"] = abduct_deg_l
        self.info_map["uarm_rot_deg_l"] = rot_deg_l
        self.info_map["uarm_raised_deg_l"] = raised_deg_l
        self.info_map["uarm_flex_deg_r"] = flex_deg_r
        self.info_map["uarm_abduct_deg_r"] = abduct_deg_r
        self.info_map["uarm_rot_deg_r"] = rot_deg_r
        self.info_map["uarm_raised_deg_r"] = raised_deg_r

        return arr_score_l, arr_score_r

    def calc_lower_arms_score(self, arr_pos):
        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.ELBOW_RIGHT, :]

        flex_deg_l = self.calc_vec_deg(shoul2elb_l, elb2wrist_l)
        flex_deg_r = self.calc_vec_deg(shoul2elb_r, elb2wrist_r)

        arr_score_l = np.ones(len(arr_pos), dtype=int)
        arr_score_l[
            (flex_deg_l < self.LOWER_ARMS_SCORE_TH_1) |
            (flex_deg_l > self.LOWER_ARMS_SCORE_TH_2)
        ] = 2

        arr_score_r = np.ones(len(arr_pos), dtype=int)
        arr_score_r[
            (flex_deg_r < self.LOWER_ARMS_SCORE_TH_1) |
            (flex_deg_r > self.LOWER_ARMS_SCORE_TH_2)
        ] = 2

        self.info_map["larm_flex_deg_l"] = flex_deg_l
        self.info_map["larm_flex_deg_r"] = flex_deg_r

        return arr_score_l, arr_score_r

    def calc_wrists_twist(self, arr_pos):
        shoul2elb_l = arr_pos[:, kinect.ELBOW_LEFT, :] - arr_pos[:, kinect.SHOULDER_LEFT, :]
        shoul2elb_r = arr_pos[:, kinect.ELBOW_RIGHT, :] - arr_pos[:, kinect.SHOULDER_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.ELBOW_RIGHT, :]
        wrist2thumb_l = arr_pos[:, kinect.THUMB_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2thumb_r = arr_pos[:, kinect.THUMB_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]

        u_elb2wrist_l = self.calc_unit_vec(elb2wrist_l)
        u_elb2wrist_r = self.calc_unit_vec(elb2wrist_r)
        u_wrist2thumb_l = self.calc_unit_vec(wrist2thumb_l)
        u_wrist2thumb_r = self.calc_unit_vec(wrist2thumb_r)
        arr_inner_l = np.sum(u_wrist2thumb_l * u_elb2wrist_l, axis=1, keepdims=True)
        arr_inner_r = np.sum(u_wrist2thumb_r * u_elb2wrist_r, axis=1, keepdims=True)
        wrist2thumb_l_proj = u_wrist2thumb_l - arr_inner_l * u_elb2wrist_l
        wrist2thumb_r_proj = u_wrist2thumb_r - arr_inner_r * u_elb2wrist_r

        u_elb2shoul_l = self.calc_unit_vec(-shoul2elb_l)
        u_elb2shoul_r = self.calc_unit_vec(-shoul2elb_r)
        arr_inner_l = np.sum(u_elb2shoul_l * u_elb2wrist_l, axis=1, keepdims=True)
        arr_inner_r = np.sum(u_elb2shoul_r * u_elb2wrist_l, axis=1, keepdims=True)
        elb2shoul_l_proj = u_elb2shoul_l - arr_inner_l * u_elb2wrist_l
        elb2shoul_r_proj = u_elb2shoul_r - arr_inner_r * u_elb2wrist_r

        # avoid zero vector
        wrist2thumb_l_proj[np.all(wrist2thumb_l_proj == 0, axis=1)] = np.array([0, 1, 0])
        wrist2thumb_r_proj[np.all(wrist2thumb_r_proj == 0, axis=1)] = np.array([0, 1, 0])

        elb2shoul_l_proj[np.all(elb2shoul_l_proj == 0, axis=1)] = np.array([0, 1, 0])
        elb2shoul_r_proj[np.all(elb2shoul_r_proj == 0, axis=1)] = np.array([0, 1, 0])

        twist_deg_l = self.calc_vec_deg(wrist2thumb_l_proj, elb2shoul_l_proj)
        twist_deg_r = self.calc_vec_deg(wrist2thumb_r_proj, elb2shoul_r_proj)

        return twist_deg_l, twist_deg_r

    def calc_wrists_score(self, arr_pos):
        wrist2hand_l = arr_pos[:, kinect.HAND_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2hand_r = arr_pos[:, kinect.HAND_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]
        wrist2thumb_l = arr_pos[:, kinect.THUMB_LEFT, :] - arr_pos[:, kinect.WRIST_LEFT, :]
        wrist2thumb_r = arr_pos[:, kinect.THUMB_RIGHT, :] - arr_pos[:, kinect.WRIST_RIGHT, :]
        elb2wrist_l = arr_pos[:, kinect.WRIST_LEFT, :] - arr_pos[:, kinect.ELBOW_LEFT, :]
        elb2wrist_r = arr_pos[:, kinect.WRIST_RIGHT, :] - arr_pos[:, kinect.ELBOW_RIGHT, :]

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

        twist_deg_l, twist_deg_r = self.calc_wrists_twist(arr_pos)

        arr_score_l = np.ones(len(arr_pos), dtype=int)
        arr_score_l[flex_deg_l > self.WRISTS_SCORE_TH] = 2
        arr_score_l[
            (twist_deg_l > self.WRISTS_TWIST_TH) |
            (deviate_deg_l > self.WRISTS_DEVIATE_TH)
        ] += 1

        arr_score_r = np.ones(len(arr_pos), dtype=int)
        arr_score_r[flex_deg_r > self.WRISTS_SCORE_TH] = 2
        arr_score_r[
            (twist_deg_r > self.WRISTS_TWIST_TH) |
            (deviate_deg_r > self.WRISTS_DEVIATE_TH)
        ] += 1

        self.info_map["wrist_flex_deg_l"] = flex_deg_l
        self.info_map["wrist_twist_deg_l"] = twist_deg_l
        self.info_map["wrist_deviate_deg_l"] = deviate_deg_l
        self.info_map["wrist_flex_deg_r"] = flex_deg_r
        self.info_map["wrist_twist_deg_r"] = twist_deg_r
        self.info_map["wrist_deviate_deg_r"] = deviate_deg_r

        return arr_score_l, arr_score_r

    def get_scores_a(self, trunk_scores, neck_scores, leg_scores, load_force=0, shock_rapid=0):
        return \
            self.table_a[trunk_scores - 1, neck_scores - 1, leg_scores - 1] + load_force + shock_rapid

    def get_scores_b(self, uarm_scores, larm_scores, wrist_scores, coupling=0):
        return \
            self.table_b[uarm_scores - 1, larm_scores - 1, wrist_scores - 1] + coupling

    def get_scores_c(self, scores_a, scores_b):
        return self.table_c[scores_a - 1, scores_b - 1]

    def run(self, save_dir, arr_pos, load_force=0, shock_rapid=0, coupling=0, activity_score=0):
        u_vert = self.estimate_vert(arr_pos)
        trunk_scores = self.calc_trunk_score(arr_pos, u_vert)
        neck_scores = self.calc_neck_score(arr_pos)
        leg_scores = self.calc_legs_score(arr_pos, u_vert)
        shoul_base = self.estimate_shoulder_base(arr_pos)
        uarm_scores_l, uarm_scores_r = self.calc_upper_arms_score(arr_pos, shoul_base)
        larm_scores_l, larm_scores_r = self.calc_lower_arms_score(arr_pos)
        wrist_scores_l, wrist_scores_r = self.calc_wrists_score(arr_pos)

        uarm_scores = np.max([uarm_scores_l, uarm_scores_r], axis=0)
        larm_scores = np.max([larm_scores_l, larm_scores_r], axis=0)
        wrist_scores = np.max([wrist_scores_l, wrist_scores_r], axis=0)

        scores_a = self.get_scores_a(trunk_scores, neck_scores, leg_scores,
                                     load_force=load_force, shock_rapid=shock_rapid)
        scores_b = self.get_scores_b(uarm_scores, larm_scores, wrist_scores,
                                     coupling=coupling)
        scores_c = self.get_scores_c(scores_a, scores_b)
        reba_scores = scores_c + activity_score

        np.savetxt(f"{save_dir}/reba_scores.csv", reba_scores, fmt="%d", delimiter=",")

        part_scores = np.stack(
            [trunk_scores, neck_scores, leg_scores,
             uarm_scores_l, uarm_scores_r,
             larm_scores_l, larm_scores_r,
             wrist_scores_l, wrist_scores_r],
            axis=1
        )

        df_part_scores = pd.DataFrame(
            part_scores,
            columns=["Trunk", "Neck", "Legs",
                     "Left_upper_arm", "Right_upper_arm",
                     "Left_lower_arm", "Right_upper_arm",
                     "Left_wrist", "Right_wrist"]
        )
        df_part_scores.to_csv(f"{save_dir}/part_scores.csv", index=False)

        pd.DataFrame.from_dict(self.info_map).to_csv(f"{save_dir}/info.csv")

        print("Done REBA scoring")

        return reba_scores


# code sample
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=1)
    parser.add_argument("--fps",
                        help="fps",
                        type=int, default=30)
    args = parser.parse_args()

    reba_ins = Reba(args.fps)
    _, _, arr_pos = kinect.read_time_ori_pos(args.input_json, args.body_id)

    name = os.path.splitext(os.path.basename(args.input_json))[0]
    save_dir = f"./outputs/reba/{name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    reba_ins.run(save_dir, arr_pos)
