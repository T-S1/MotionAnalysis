# import pdb; pdb.set_trace()
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import src.kinect as kinect

OUTPUT_DIR = "./dtw"


def trans_coord(arr_pos):

    arr_pos_trans = np.zeros_like(arr_pos)
    pel = arr_pos[:, kinect.PELVIS, :]
    nav = arr_pos[:, kinect.SPINE_NAVAL, :]
    hip_l = arr_pos[:, kinect.HIP_LEFT, :]
    hip_r = arr_pos[:, kinect.HIP_RIGHT, :]

    e_x = hip_r - hip_l
    e_x /= np.linalg.norm(e_x, ord=2, axis=1, keepdims=True)
    e_y = np.cross(nav - pel, e_x)
    e_y /= np.linalg.norm(e_y, ord=2, axis=1, keepdims=True)
    e_z = np.cross(e_x, e_y)
    rot_mat = np.stack([e_x, e_y, e_z], axis=1)     # (T, 3, 3)
    rot_mat = np.expand_dims(rot_mat, axis=1)

    arr_pos_temp = np.expand_dims(arr_pos, axis=3)      # (T, J, 3, 1)
    arr_pos_temp = np.matmul(rot_mat, arr_pos_temp)
    arr_pos_temp = np.squeeze(arr_pos_temp)             # (T, J, 3)
    pel_trans = arr_pos_temp[:, kinect.PELVIS, :]       # (T, 3)
    arr_pos_temp -= np.expand_dims(pel_trans, axis=1)

    arr_pos_trans[:, :, :] = arr_pos_temp

    return arr_pos_trans


class MotionDrawer():
    def __init__(self, arr_dist, path_list):

        self.temp_dir = "./temp"
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.dim_x = 0
        self.dim_y = 1
        self.dim_z = 2
        self.alpha = 1000    # [mm]
        self.width = 512
        self.height = 512

        self.arr_dist = arr_dist
        self.path_list = path_list

        self.fig = plt.figure(tight_layout=True)
        self.ax = self.fig.add_subplot(projection="3d")

    def draw_frame(self, arr_pos, idx):

        pel = arr_pos[idx, kinect.PELVIS]
        self.ax.set(xlim3d=(pel[self.dim_x] - self.alpha, pel[self.dim_x] + self.alpha))
        self.ax.set(ylim3d=(pel[self.dim_y] - self.alpha, pel[self.dim_y] + self.alpha))
        self.ax.set(zlim3d=(pel[self.dim_z] - self.alpha, pel[self.dim_z] + self.alpha))

        for joint_1, joint_2 in kinect.bone_list:
            pos_1 = arr_pos[idx, joint_1]
            pos_2 = arr_pos[idx, joint_2]
            self.ax.plot(
                [pos_1[self.dim_x], pos_2[self.dim_x]],
                [pos_1[self.dim_y], pos_2[self.dim_y]],
                [pos_1[self.dim_z], pos_2[self.dim_z]],
                color="blue", markersize=1
            )

        plt.savefig(f"{self.temp_dir}/temp.jpg")
        self.ax.cla()

        im = cv2.imread(f"{self.temp_dir}/temp.jpg")
        if np.shape(im)[0] < np.shape(im)[1]:
            width_temp = round(im.shape[1] * self.height / im.shape[0])
            im = cv2.resize(im, (width_temp, self.height))
            left = (self.width - self.height) // 2
            right = (self.width + self.height) // 2
            im = im[:, left: right]
        else:
            height_temp = round(im.shape[0] * self.width / im.shape[1])
            im = cv2.resize(im, (self.width, height_temp))
            top = (self.height - self.width) // 2
            bottom = (self.height + self.width) // 2
            im = im[top: bottom, :]

        return im

    def vis_warped_motion(self, arr_pos_1, arr_pos_2, output_mp4, fps, target_joint=kinect.WRIST_RIGHT):

        print("Processing")

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(output_mp4, fourcc, fps, (self.width * 2, self.height))

        for i, j in tqdm(self.path_list[target_joint]):

            im_1 = self.draw_frame(arr_pos_1, i)
            im_2 = self.draw_frame(arr_pos_2, j)

            im = np.concatenate([im_1, im_2], axis=1)

            cv2.putText(
                im, f"(i, j) = ({i}, {j})", (0, self.height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_4
            )

            writer.write(im)

        writer.release()


class DTWDrawer():
    def __init__(self, arr_dist, path_list):

        self.arr_dist = arr_dist
        self.path_list = path_list

        self.fig, self.ax = plt.subplots(tight_layout=True)

    def vis_scores(self, output_jpg):

        self.ax.barh(kinect.JOINT_NAME_LIST, self.arr_dist, alpha=0)
        plt.savefig(output_jpg)
        plt.cla()

    def vis_paths(self, out_jpg_dir):

        i_max = self.path_list[0][-1][0]
        j_max = self.path_list[0][-1][1]
        x = np.zeros((i_max + 1, j_max + 1))

        for joint in range(kinect.NUM_JOINTS):
            x[:, :] = 0
            joint_name = kinect.get_joint_name(joint)
            output_jpg = f"{out_jpg_dir}/{joint_name}.jpg"
            for i, j in self.path_list[joint]:
                x[i, j] = 1
            self.ax.imshow(x)
            self.ax.set_xlabel("j")
            self.ax.set_ylabel("i")
            plt.savefig(output_jpg)
            plt.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json_1",
                        help="input json file path 1",
                        type=str)
    parser.add_argument("input_json_2",
                        help="input json file path 2",
                        type=str)
    parser.add_argument("-id_1", "--body_id_1",
                        help="target body id 1",
                        type=int, default=1)
    parser.add_argument("-id_2", "--body_id_2",
                        help="target body id 2",
                        type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    arr_time_1, _, arr_pos_1 = kinect.read_time_ori_pos(args.input_json_1, args.body_id_1)   # (T, J, D)
    arr_time_2, _, arr_pos_2 = kinect.read_time_ori_pos(args.input_json_2, args.body_id_2)

    arr_pos_trans_1 = trans_coord(arr_pos_1)
    arr_pos_trans_2 = trans_coord(arr_pos_2)

    dist_list = []
    path_list = []

    for j in range(kinect.NUM_JOINTS):
        dist, path = fastdtw(arr_pos_trans_1[:, j, :], arr_pos_trans_2[:, j, :], dist=euclidean)
        dist_list.append(dist)
        path_list.append(path)

    name_1 = os.path.splitext(os.path.basename(args.input_json_1))[0]
    name_2 = os.path.splitext(os.path.basename(args.input_json_2))[0]
    name = f"{name_1}-{name_2}"
    output_json = f"{OUTPUT_DIR}/{name}.json"

    with open(output_json, "w") as f:
        json.dump(
            {"dist": dist_list,
             "path": path_list},
            f, indent=4
        )

    dtw_drawer = DTWDrawer(dist_list, path_list)

    scores_jpg = f"{OUTPUT_DIR}/scores.jpg"
    dtw_drawer.vis_scores(scores_jpg)

    path_jpg_dir = f"{OUTPUT_DIR}/paths"
    if not os.path.isdir(path_jpg_dir):
        os.makedirs(path_jpg_dir)

    dtw_drawer.vis_paths(path_jpg_dir)

    dif_time = arr_time_1[1:] - arr_time_1[:-1]
    mean_dif = np.mean(dif_time) / 1e6
    fps = 1 / mean_dif

    motion_drawer = MotionDrawer(dist_list, path_list)
    motion_drawer.vis_warped_motion(arr_pos_1, arr_pos_2, f"{OUTPUT_DIR}/compare.mp4", fps)

    print("Done")


if __name__ == "__main__":
    main()
