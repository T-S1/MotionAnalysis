import os
import atexit
import tqdm

import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import kinect as kinect


class TextProp():
    FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FONT_SCALE = 1
    THICKNESS = 1
    COLOR = (0, 0, 0)
    LINE_TYPE = cv2.LINE_4


class MotionDrawer():
    def __init__(
        self,
        out_mp4,
        fps=30,
        alpha=1000,
        size=(512, 512),
        dim_list=[0, 1, 2],
        quiv_len=50,
        arrow_length_ratio=0.1,
        linewidth=0.5
    ):
        self.temp_dir = "./temp"
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.fps = fps
        self.alpha = alpha   # [mm]
        self.size = size
        self.dim_x = dim_list[0]
        self.dim_y = dim_list[1]
        self.dim_z = dim_list[2]
        self.quiv_len = quiv_len
        self.line_width = linewidth
        self.bone_list = np.array(kinect.bone_list)

        self.out_mp4 = out_mp4

        fig = plt.figure()
        self.ax = fig.add_subplot(projection="3d")

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.writer = cv2.VideoWriter(self.mp4, fourcc, self.fps, self.size)
        atexit.register(self.cleanup)

        self.frame = np.zeros(size)

    def cleanup(self):
        self.writer.release()

    def set_lims(self, center):
        self.ax.set(xlim3d=(center[self.dim_x] - self.alpha,
                    center[self.dim_x] + self.alpha))
        self.ax.set(ylim3d=(center[self.dim_y] - self.alpha,
                    center[self.dim_y] + self.alpha))
        self.ax.set(zlim3d=(center[self.dim_z] - self.alpha,
                    center[self.dim_z] + self.alpha))

    def add_positions(self, positions):

        x = positions[self.bone_list[:, 0], self.dim_x]
        y = positions[self.bone_list[:, 0], self.dim_y]
        z = positions[self.bone_list[:, 0], self.dim_z]

        u = positions[self.bone_list[:, 1], self.dim_x] - x
        v = positions[self.bone_list[:, 1], self.dim_y] - y
        w = positions[self.bone_list[:, 1], self.dim_z] - z

        self.ax.scatter(x, y, z, c="black", ms=1)
        self.ax.quiver(
            x, y, z, u, v, w,
            arrow_length_ratio=0, c="black", lw=0.5
        )

    def rot_coords(self, orientations):
        qs_rot = quaternion.as_quat_array(orientations)     # (joint, wxyz)
        qs_coords = quaternion.from_vector_part(np.eye(3))  # (dim, wxyz)
        qs_rot_conj = np.conjugate(qs_rot)                  # (joint, wxyz)
        q_u = qs_rot * qs_coords[0] * qs_rot_conj           # (joint, wxyz)
        q_v = qs_rot * qs_coords[1] * qs_rot_conj
        q_w = qs_rot * qs_coords[2] * qs_rot_conj
        return quaternion.as_vector_part([q_u, q_v, q_w])   # (uvw, joint, dim)

    def add_orientaions(self, positions, orientations):

        x = positions[:, self.dim_x]
        y = positions[:, self.dim_y]
        z = positions[:, self.dim_z]
        u, v, w = self.rot_coords(orientations) * self.quiv_len

        self.ax.quiver(
            x, y, z, u[:, self.dim_x], u[:, self.dim_y], u[:, self.dim_z],
            arrow_length_ratio=0.1, color="red", linewidth=0.5
        )
        self.ax.quiver(
            x, y, z, v[:, self.dim_x], v[:, self.dim_y], v[:, self.dim_z],
            arrow_length_ratio=0.1, color="green", linewidth=0.5
        )
        self.ax.quiver(
            x, y, z, w[:, self.dim_x], w[:, self.dim_y], w[:, self.dim_z],
            arrow_length_ratio=0.1, color="blue", linewidth=0.5
        )

    def draw_frame(self):
        plt.savefig(f"{self.temp_dir}/temp.jpg")
        self.ax.cla()

        im = cv2.imread(f"{self.temp_dir}/temp.jpg")
        self.frame = cv2.resize(im, self.size)

    def add_text(self, text, row=0):
        (w, h), baseline = cv2.getTextSize(
            text, TextProp.FONT_FACE, TextProp.FONT_SCALE, TextProp.THICKNESS
        )
        left = baseline
        left_bottom = (h + 2 * baseline) * (row + 1)
        cv2.putText(
            self.frame, text, (left, left_bottom),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(0, 0, 0),
            TextProp.THICKNESS, TextProp.LINE_TYPE
        )

    def write_frame(self):
        self.writer.write(self.frame)

    def run(self, arr_pos, arr_ori=None):
        for t in tqdm(range(len(arr_pos))):
            pel = self.arr_pos[t, kinect.PELVIS]

            self.set_lims(pel)
            self.add_positions(arr_pos)

            if arr_ori is not None:
                self.add_orientaions(arr_pos, arr_ori)

            self.write()
