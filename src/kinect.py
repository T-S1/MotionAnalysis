import json
import numpy as np

NUM_JOINTS = 32

[
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

bone_list = [
    [
        SPINE_CHEST,
        SPINE_NAVAL
    ],
    [
        SPINE_NAVAL,
        PELVIS
    ],
    [
        SPINE_CHEST,
        NECK
    ],
    [
        NECK,
        HEAD
    ],
    [
        HEAD,
        NOSE
    ],
    [
        SPINE_CHEST,
        CLAVICLE_LEFT
    ],
    [
        CLAVICLE_LEFT,
        SHOULDER_LEFT
    ],
    [
        SHOULDER_LEFT,
        ELBOW_LEFT
    ],
    [
        ELBOW_LEFT,
        WRIST_LEFT
    ],
    [
        WRIST_LEFT,
        HAND_LEFT
    ],
    [
        HAND_LEFT,
        HANDTIP_LEFT
    ],
    [
        WRIST_LEFT,
        THUMB_LEFT
    ],
    [
        PELVIS,
        HIP_LEFT
    ],
    [
        HIP_LEFT,
        KNEE_LEFT
    ],
    [
        KNEE_LEFT,
        ANKLE_LEFT
    ],
    [
        ANKLE_LEFT,
        FOOT_LEFT
    ],
    [
        NOSE,
        EYE_LEFT
    ],
    [
        EYE_LEFT,
        EAR_LEFT
    ],
    [
        SPINE_CHEST,
        CLAVICLE_RIGHT
    ],
    [
        CLAVICLE_RIGHT,
        SHOULDER_RIGHT
    ],
    [
        SHOULDER_RIGHT,
        ELBOW_RIGHT
    ],
    [
        ELBOW_RIGHT,
        WRIST_RIGHT
    ],
    [
        WRIST_RIGHT,
        HAND_RIGHT
    ],
    [
        HAND_RIGHT,
        HANDTIP_RIGHT
    ],
    [
        WRIST_RIGHT,
        THUMB_RIGHT
    ],
    [
        PELVIS,
        HIP_RIGHT
    ],
    [
        HIP_RIGHT,
        KNEE_RIGHT
    ],
    [
        KNEE_RIGHT,
        ANKLE_RIGHT
    ],
    [
        ANKLE_RIGHT,
        FOOT_RIGHT
    ],
    [
        NOSE,
        EYE_RIGHT
    ],
    [
        EYE_RIGHT,
        EAR_RIGHT
    ]
]

JOINT_NAME_LIST = [
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
]


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


def get_joint_name(joint_id):
    return JOINT_NAME_LIST[joint_id]
