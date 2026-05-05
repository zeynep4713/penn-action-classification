import os
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_keypoints(frame, keypoints):
    h, w = frame.shape[:2]

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    for connection in SKELETON_CONNECTIONS:
        i, j = connection
        if keypoints[i][2] > 0.1 and keypoints[j][2] > 0.1:
            pt1 = (int(keypoints[i][0] * w), int(keypoints[i][1] * h))
            pt2 = (int(keypoints[j][0] * w), int(keypoints[j][1] * h))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2, lineType=cv2.LINE_AA)

def get_video_resolution_from_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    dims = mat["dimensions"].flatten()
    height, width = int(dims[0]), int(dims[1])
    return width, height

def visualize_and_save_video(keypoints_dir, frame_root_dir, annotation_dir, output_videos_dir):
    os.makedirs(output_videos_dir, exist_ok=True)

    for vid_file in tqdm(os.listdir(keypoints_dir)):
        vid_name = os.path.splitext(vid_file)[0]
        keypoints_path = os.path.join(keypoints_dir, vid_file)
        frame_folder = os.path.join(frame_root_dir, vid_name)
        annotation_path = os.path.join(annotation_dir, f"{vid_name}.mat")

        try:
            keypoints_seq = np.load(keypoints_path)  # shape: (T, 17, 3)
        except:
            continue

        if not os.path.exists(annotation_path):
            print(f"Annotation file missing for {vid_name}")
            continue

        frame_files = sorted(os.listdir(frame_folder))
        if len(frame_files) == 0 or len(keypoints_seq) == 0:
            continue

        FRAME_WIDTH, FRAME_HEIGHT = get_video_resolution_from_mat(annotation_path)

        output_path = os.path.join(output_videos_dir, f"{vid_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 10, (FRAME_WIDTH, FRAME_HEIGHT))

        for i, frame_file in enumerate(frame_files):
            if i >= len(keypoints_seq):
                break
            frame_path = os.path.join(frame_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            draw_keypoints(frame, keypoints_seq[i])
            writer.write(frame)

        writer.release()

visualize_and_save_video(
    keypoints_dir="Penn_Action/keypoints_yl",
    frame_root_dir="Penn_Action/frames",
    annotation_dir="Penn_Action/labels",
    output_videos_dir="Penn_Action/yolo_videos"
)
