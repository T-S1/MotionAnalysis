import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


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


def read_time_ori_pos(input_json, target_body_id):

    with open(input_json, "r") as f:
        track_map = json.load(f)

    timestamp_list = []
    orientations_list = []
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

                joint_orientations = body["joint_orientations"]
                joint_positions = body["joint_positions"]

                timestamp_list.append(timestamp_usec)
                orientations_list.append(joint_orientations)
                positions_list.append(joint_positions)

    num_frames = len(timestamp_list)

    arr_timestamp = np.zeros(num_frames)
    arr_orientations = np.zeros((num_frames, NUM_JOINTS, 4))
    arr_positions = np.zeros((num_frames, NUM_JOINTS, 3))

    arr_timestamp[:] = np.array(timestamp_list)
    arr_orientations[:, :, :] = np.array(orientations_list)
    arr_positions[:, :, :] = np.array(positions_list)         # (T, J, 3)

    return arr_timestamp, arr_orientations, arr_positions


class MotionDrawer():
    def __init__(self, arr_positions, arr_orientations, mp4, fps):
        self.temp_dir = "./temp"
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.alpha = 1000   # [mm]
        self.size = (512, 512)

        self.mp4 = mp4
        self.fps = fps

        self.arr_pos = arr_positions
        self.arr_ori = arr_orientations

        fig = plt.figure()
        self.ax = fig.add_subplot(projection="3d")

    def calc_pose(self, t, j, k=50):
        q = self.arr_ori[t, j]
        quat = np.quaternion(q[0], q[1], q[2], q[3])

        coord = np.eye(3)
        x = quaternion.rotate_vectors(quat, coord[0])
        y = quaternion.rotate_vectors(quat, coord[1])
        z = quaternion.rotate_vectors(quat, coord[2])

        return x * k, y * k, z * k

    def run(self):
        print("processing")

        for t in tqdm(range(len(self.arr_pos))):
            pel = self.arr_pos[t, PELVIS]
            self.ax.set(xlim3d=(pel[0] - self.alpha, pel[0] + self.alpha))
            self.ax.set(ylim3d=(pel[2] - self.alpha, pel[2] + self.alpha))
            self.ax.set(zlim3d=(pel[1] - self.alpha, pel[1] + self.alpha))
            for j in range(NUM_JOINTS):
                pos = self.arr_pos[t, j]

                base_x, base_y, base_z = self.calc_pose(t, j)

                self.ax.quiver(
                    pos[0], pos[2], -pos[1],
                    base_x[0], base_x[2], -base_x[1],
                    arrow_length_ratio=0.1,
                    color="red",
                    linewidth=0.5
                )
                self.ax.quiver(
                    pos[0], pos[2], -pos[1],
                    base_y[0], base_y[2], -base_y[1],
                    arrow_length_ratio=0.1,
                    color="green",
                    linewidth=0.5
                )
                self.ax.quiver(
                    pos[0], pos[2], -pos[1],
                    base_z[0], base_z[2], -base_z[1],
                    arrow_length_ratio=0.1,
                    color="blue",
                    linewidth=0.5
                )

            plt.savefig(f"{self.temp_dir}/{t:010}.jpg")
            self.ax.cla()

        print("saving")

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(self.mp4, fourcc, self.fps, self.size)

        for i in tqdm(range(len(self.arr_pos))):
            im = cv2.imread(f"{self.temp_dir}/{i:010}.jpg")
            im = cv2.resize(im, self.size)
            writer.write(im)

            os.remove(f"{self.temp_dir}/{i:010}.jpg")

        writer.release()

        print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json",
                        help="input json file path",
                        type=str)
    parser.add_argument("output_mp4",
                        help="output mp4 file path",
                        type=str)
    parser.add_argument("-id", "--body_id",
                        help="target body id",
                        type=int, default=1)
    args = parser.parse_args()

    arr_timestamp, arr_orientations, arr_positions \
        = read_time_ori_pos(args.input_json, args.body_id)

    dif_time = arr_timestamp[1:] - arr_timestamp[:-1]
    mean_dif = np.mean(dif_time) / 1e6
    fps = 1 / mean_dif
    print(fps, "fps")

    motion_drawer = MotionDrawer(arr_positions, arr_orientations, args.output_mp4, fps)
    motion_drawer.run()


if __name__ == "__main__":
    main()
