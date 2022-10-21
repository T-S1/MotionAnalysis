import argparse
import os
import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import src.kinect as kinect
from src.reba import Reba


class MotionDrawer():
    def __init__(self):

        self.temp_dir = "./temp"
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.dim_x = 0
        self.dim_y = 2
        self.dim_z = 1
        self.alpha = 1000    # [mm]
        self.trim_cent_gap = 128
        self.side_length = 512

        # self.path_list = path_list

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(projection="3d")

    def draw_frame(self, arr_pos, idx, color):

        pel = arr_pos[idx, kinect.PELVIS]
        self.ax.set(xlim3d=(pel[self.dim_x] - self.alpha, pel[self.dim_x] + self.alpha))
        self.ax.set(ylim3d=(pel[self.dim_y] - self.alpha, pel[self.dim_y] + self.alpha))
        self.ax.set(zlim3d=(pel[self.dim_z] + self.alpha, pel[self.dim_z] - self.alpha))

        self.ax.scatter(
            arr_pos[idx, :, self.dim_x],
            arr_pos[idx, :, self.dim_y],
            arr_pos[idx, :, self.dim_z],
            color=plt.get_cmap("Oranges")((color - 1) * 18)
        )

        for joint_1, joint_2 in kinect.bone_list:
            pos_1 = arr_pos[idx, joint_1]
            pos_2 = arr_pos[idx, joint_2]
            self.ax.plot(
                [pos_1[self.dim_x], pos_2[self.dim_x]],
                [pos_1[self.dim_y], pos_2[self.dim_y]],
                [pos_1[self.dim_z], pos_2[self.dim_z]],
                color=plt.get_cmap("Oranges")((color - 1) * 18)
            )

        plt.savefig(f"{self.temp_dir}/temp.jpg")
        self.ax.cla()

        im = cv2.imread(f"{self.temp_dir}/temp.jpg")
        im = cv2.resize(im, (self.side_length, self.side_length))

        return im

    def vis_motion(self, arr_pos, output_mp4, fps,
                   arr_color=None, target_joint=kinect.WRIST_RIGHT):

        print("Processing")

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(output_mp4, fourcc, fps, (self.side_length, self.side_length))

        for i in tqdm.tqdm(range(len(arr_pos))):
            im = self.draw_frame(arr_pos, i, color=arr_color[i])
            
            text = (f"Score: {arr_color[i]}")
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
    save_dir = f"./reba/{name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    reba_scores = reba_ins.run(save_dir, arr_pos)

    MotionDrawer().vis_motion(arr_pos, "./reba/result.mp4", 30, reba_scores)


if __name__ == "__main__":
    main()