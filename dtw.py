import argparse
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import src.reader as reader


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

    _, _, arr_pos_1 = reader.read_time_ori_pos(args.input_json_1, args.body_id_1)   # (T, J, D)
    _, _, arr_pos_2 = reader.read_time_ori_pos(args.input_json_2, args.body_id_2)

    dist, path = fastdtw(arr_pos_1, arr_pos_2)

    


if __name__ == "__main__":
    main()
