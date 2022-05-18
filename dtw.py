# import pdb; pdb.set_trace()
import argparse
import os
from typing import List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import src.kinect as kinect

OUTPUT_DIR = "./dtw"


def trans_coord(arr_pos):

    arr_pos_trans = np.zeros_like(arr_pos)
    pel = arr_pos[:, kinect.PELVIS, :]
    hip_l = arr_pos[:, kinect.HIP_LEFT, :]
    hip_r = arr_pos[:, kinect.HIP_RIGHT, :]
    hip_l2r = hip_r - hip_l

    for joint in range(kinect.NUM_JOINTS):
        arr_pos_trans[:, joint, :] = arr_pos[:, joint, :] - pel

    return arr_pos_trans


class Visualizer():
    def __init__(self):
        fig, self.ax = plt.subplots(
            tight_layout=True
        )

    def vis_scores(self, output_jpg: str, arr_dist: np.ndarray) -> None:

        self.ax.bar(kinect.JOINT_NAME_LIST, arr_dist)
        plt.savefig(output_jpg)
        plt.cla()

    def vis_paths(self, out_jpg_dir: str, path_list: List[List[int]]) -> None:

        i_max = path_list[0][-1][0]
        j_max = path_list[0][-1][1]
        x = np.zeros((i_max + 1, j_max + 1))

        for joint in range(kinect.NUM_JOINTS):
            x[:, :] = 0
            joint_name = kinect.get_joint_name(joint)
            output_jpg = f"{out_jpg_dir}/{joint_name}.jpg"
            for i, j in path_list[joint]:
                x[i, j] = 1
            self.ax.imshow(x)
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

    name_1 = os.path.splitext(os.path.basename(args.input_json_1))[0]
    name_2 = os.path.splitext(os.path.basename(args.input_json_2))[0]
    name = f"{name_1}-{name_2}"
    output_json = f"{OUTPUT_DIR}/{name}.json"

    dist_list = []
    path_list = []

    if not os.path.isfile(output_json):

        _, _, arr_pos_1 = kinect.read_time_ori_pos(args.input_json_1, args.body_id_1)   # (T, J, D)
        _, _, arr_pos_2 = kinect.read_time_ori_pos(args.input_json_2, args.body_id_2)

        arr_pos_1 = trans_coord(arr_pos_1)
        arr_pos_2 = trans_coord(arr_pos_2)

        for j in range(kinect.NUM_JOINTS):
            dist, path = fastdtw(arr_pos_1[:, j, :], arr_pos_2[:, j, :], dist=euclidean)
            dist_list.append(dist)
            path_list.append(path)

        with open(output_json, "w") as f:
            json.dump(
                {"dist": dist_list,
                 "path": path_list},
                f, indent=4
            )
    else:
        with open(output_json, "r") as f:
            dtw_map = json.load(f)

        dist_list = dtw_map["dist"]
        path_list = dtw_map["path"]

    visualizer = Visualizer()

    scores_jpg = f"{OUTPUT_DIR}/scores.jpg"
    visualizer.vis_scores(scores_jpg, dist_list)

    path_jpg_dir = f"{OUTPUT_DIR}/paths"
    if not os.path.isdir(path_jpg_dir):
        os.makedirs(path_jpg_dir)

    visualizer.vis_paths(path_jpg_dir, path_list)

    print("Done")


if __name__ == "__main__":
    main()
