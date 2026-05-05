import os
import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from ultralytics import YOLO

from scipy.signal import savgol_filter

# Load YOLOv8 pose model
model = YOLO("yolov8x-pose.pt")

# Indices of 17 body keypoints from YOLOv8
KEYPOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def get_pose_distance(pose1, pose2, conf_weighted=True):
    if conf_weighted:
        conf = (pose1[:, 2] + pose2[:, 2]) / 2
        conf[conf < 0.1] = 0.1       # Prevent divison by 0
        return np.mean(np.linalg.norm(pose1[:, :2] - pose2[:, :2], axis=1) / conf)
    else:
        return np.mean(np.linalg.norm(pose1[:, :2] - pose2[:, :2], axis=1))

def extract_keypoints_from_images(folder, width, height, max_dist_threshold=1500, min_visible=8):
    frames = sorted(os.listdir(folder))
    keypoints_seq = []
    last_pose = None
    for idx, fname in enumerate(frames):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        results = model(img)
        kpts = results[0].keypoints

        selected_k = np.zeros((17, 3))  # default: blank keypoints

        if kpts is not None and kpts.data.shape[0] > 0:
            all_poses = kpts.data.cpu().numpy()  # (num_people, 17, 3)

            # Normalize for comparison
            for i in range(all_poses.shape[0]):
                all_poses[i, :, 0] /= width
                all_poses[i, :, 1] /= height

            if idx == 0 or last_pose is None:
                # First frame: pick person closest to center
                image_center = np.array([0.5, 0.5])  # normalized
                centers = all_poses[:, :, :2].mean(axis=1)
                dists = np.linalg.norm(centers - image_center, axis=1)
                best_idx = int(np.argmin(dists))
                selected_k = all_poses[best_idx]
                last_pose = selected_k
            else:
                # Compare all to last_pose
                dists = [get_pose_distance(pose, last_pose) for pose in all_poses]
                visibilities = [np.sum(p[:, 2] > 0.3) for p in all_poses]

                # Mask out poses with too few visible keypoints
                valid_idxs = [i for i, v in enumerate(visibilities) if v >= min_visible]

                if valid_idxs:
                    best_valid_idx = min(valid_idxs, key=lambda i: dists[i])
                    if dists[best_valid_idx] < max_dist_threshold:
                        selected_k = all_poses[best_valid_idx]
                        last_pose = selected_k
                    else:
                        # fallback: reuse previous pose for continuity
                        selected_k = last_pose
                else:
                    selected_k = last_pose
        else:
            selected_k = last_pose if last_pose is not None else np.zeros((17, 3))

        keypoints_seq.append(selected_k)

    keypoints_seq = np.array(keypoints_seq)
    keypoints_seq = smooth_keypoints_sequence(keypoints_seq, window_size=5, polyorder=2)
    return keypoints_seq

def smooth_keypoints_sequence(keypoints_seq, window_size=5, polyorder=2):
    smoothed = np.copy(keypoints_seq)
    T, N, C = smoothed.shape

    for i in range(N):
        for j in range(2):  # x and y only
            if T >= window_size:
                smoothed[:, i, j] = savgol_filter(
                    keypoints_seq[:, i, j],
                    window_length=window_size,
                    polyorder=polyorder,
                    mode='interp'
                )
    return smoothed

def extract_keypoints_dataset(data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_ids = sorted(os.listdir(data_root))

    for video_id in tqdm(video_ids):
        folder = os.path.join(data_root, video_id)
        mat_path = os.path.join("Penn_Action/labels", video_id + ".mat")
        if not os.path.exists(folder) or not os.path.exists(mat_path):
            continue

        try:
            mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            if 'action' not in mat or 'dimensions' not in mat:
                print(f"Missing keys in {video_id}")
                continue

            height, width, _ = mat['dimensions']
            keypoints = extract_keypoints_from_images(folder, width=width, height=height)

            save_path = os.path.join(output_dir, video_id + ".npy")
            np.save(save_path, keypoints)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")

# Run extraction
extract_keypoints_dataset(
    data_root="Penn_Action/frames",
    output_dir="Penn_Action/keypoints_yx"
)
