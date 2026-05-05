import cv2
import numpy as np
import os
import glob

# Parameters
video_id = "0013"  # Change to the video you want to visualize
keypoints_path = f"Penn_Action/keypoints_yl/{video_id}.npy"     # 13, 660, 403, 1278
frames_dir = f"Penn_Action/frames/{video_id}"
visibility_threshold = 0.0  # Keypoints with visibility below this are skipped

# Load keypoints
keypoints = np.load(keypoints_path)  # Shape: (num_frames, 17, 3)

# Get sorted list of frame image paths
frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

# Loop through frames and overlay keypoints
for i, frame_path in enumerate(frame_paths):
    if i >= len(keypoints):
        break

    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]

    frame_keypoints = keypoints[i]  # Shape: (17, 3)
    for keypoint in frame_keypoints:
        x_norm, y_norm, visibility = keypoint
        if visibility < visibility_threshold:
            continue  # Skip invisible keypoints

        x = int(x_norm * w)
        y = int(y_norm * h)
        if x != 0 and y != 0:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Keypoints", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
