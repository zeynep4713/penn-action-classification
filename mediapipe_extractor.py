import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

mp_pose = mp.solutions.pose

# Indices of selected 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
SELECTED_KEYPOINT_INDICES = [
    0,   # Nose
    5,   # Left eye
    6,   # Right eye
    7,   # Left ear
    8,   # Right ear
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip
    24,  # Right hip
    25,  # Left knee
    26,  # Right knee
    27,  # Left ankle
    28   # Right ankle
]

def extract_keypoints_from_frame(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None

    keypoints = []
    for idx in SELECTED_KEYPOINT_INDICES:
        lm = results.pose_landmarks.landmark[idx]
        x = lm.x
        y = lm.y
        v = lm.visibility
        keypoints.append([x, y, v])

    return np.array(keypoints)  # shape (17, 3)

def extract_keypoints_from_video_mediapipe(frame_folder, output_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    keypoints_seq = []

    frame_files = sorted(os.listdir(frame_folder))
    for frame_name in frame_files:
        frame_path = os.path.join(frame_folder, frame_name)
        image = cv2.imread(frame_path)
        if image is None:
            continue

        keypoints = extract_keypoints_from_frame(image, pose)
        if keypoints is not None and keypoints.shape == (17, 3):
            keypoints_seq.append(keypoints)

    pose.close()

    if len(keypoints_seq) == 0:
        keypoints_seq = np.zeros((1, 17, 3))  # at least 1 frame of zeros
    else:
        keypoints_seq = np.stack(keypoints_seq)  # (T, 17, 3)

    np.save(output_path, keypoints_seq)


def batch_extract_mediapipe(frame_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for vid in tqdm(os.listdir(frame_root)):
        frame_folder = os.path.join(frame_root, vid)
        if not os.path.isdir(frame_folder):
            continue
        output_path = os.path.join(out_dir, vid + ".npy")
        extract_keypoints_from_video_mediapipe(frame_folder, output_path)

batch_extract_mediapipe("Penn_Action/frames", "Penn_Action/keypoints_m")
