import argparse
import os
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import cv2
import src.kinect as kinect
from src.reba import Reba


class MotionDrawer():
    def __init__(self):

        self.dim_x = 0
        self.dim_y = 2
        self.dim_z = 1
        self.alpha = 1000    # [mm]
        self.trim_cent_gap = 128
        self.side_length = 512

        self.fig = plt.figure(figsize=(16, 16), dpi=32)
        self.ax = self.fig.add_subplot(projection="3d")
        self.scat_col = self.ax.scatter(
            np.zeros(kinect.NUM_JOINTS),
            np.zeros(kinect.NUM_JOINTS),
            np.zeros(kinect.NUM_JOINTS),
            color="blue"
        )
        segs = [[[0, 0, 0], [0, 0, 0]] for _ in kinect.bone_list]
        line_segments = Line3DCollection(segs, colors="black")
        self.line_col = self.ax.add_collection(line_segments)

    def draw_frame(self, arr_pos, idx):

        pel = arr_pos[idx, kinect.PELVIS]
        self.ax.set(
            xlim3d=(pel[self.dim_x] - self.alpha, pel[self.dim_x] + self.alpha),
            ylim3d=(pel[self.dim_y] - self.alpha, pel[self.dim_y] + self.alpha),
            zlim3d=(pel[self.dim_z] + self.alpha, pel[self.dim_z] - self.alpha)
        )

        self.scat_col.set_offsets(arr_pos[idx, :, [self.dim_x, self.dim_y]].T)
        self.scat_col.set_3d_properties(arr_pos[idx, :, self.dim_z], "z")

        segs = [
            [
                arr_pos[idx, joint_1, [self.dim_x, self.dim_y, self.dim_z]],
                arr_pos[idx, joint_2, [self.dim_x, self.dim_y, self.dim_z]]
            ] for joint_1, joint_2 in kinect.bone_list
        ]
        self.line_col.set_segments(segs)

        self.fig.canvas.draw()
        im = np.array(self.fig.canvas.renderer.buffer_rgba())
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

        return im

    def vis_motion(self, arr_pos, output_mp4, fps,
                   arr_score=None, target_joint=kinect.WRIST_RIGHT):

        print("Processing")

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(output_mp4, fourcc, fps, (self.side_length, self.side_length))

        for i in tqdm.tqdm(range(len(arr_pos))):
            im = self.draw_frame(arr_pos, i)
            
            text = (f"Score: {arr_score[i]}")
            font_face = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1
            thickness = 1
            color = (0, 0, 0)
            line_type = cv2.LINE_4

            (w, h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            cv2.putText(
                im, text, (baseline, h + 2 * baseline),
                font_face, font_scale, color, thickness, line_type
            )

            if arr_score[i] == 1:
                color = (89, 208, 143)
            elif arr_score[i] < 4:
                color = (111, 220, 253)
            elif arr_score[i] < 8:
                color = (49, 195, 253)
            elif arr_score[i] < 11:
                color = (30, 40, 246)
            else:
                color = (21, 29, 192)
            cv2.rectangle(
                im, (w + 2 * baseline, baseline),
                (w + h + 3 * baseline, h + 2 * baseline),
                color, thickness=cv2.FILLED
            )

            writer.write(im)

        writer.release()


def main():
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
    save_dir = f"./reba"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    reba_scores = reba_ins.run(save_dir, arr_pos)

    MotionDrawer().vis_motion(arr_pos, f"{save_dir}/result.mp4", 30, reba_scores)


if __name__ == "__main__":
    main()
