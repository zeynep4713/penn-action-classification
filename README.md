# Human Action Recognition on Penn Action Dataset

An end-to-end human action recognition pipeline that uses MediaPipe and YOLOv8 for keypoint extraction, and an LSTM model for temporal classification on the Penn Action Dataset.

## Features
* **Dual Extraction**: Supports keypoint extraction via MediaPipe and YOLOv8-pose.
* **Temporal Smoothing**: Implements Savitzky-Golay filtering to reduce jitter in pose sequences.
* **PoseLSTM Model**: A PyTorch-based LSTM architecture optimized with packed sequences for variable-length video data.
* **Robust Training Loop**: Features gradient norm clipping, early stopping, and an adaptive learning rate to ensure stable convergence.
* **Visualization**: Tools for overlaying predicted skeletons and exporting results as MP4.

## Performance
* **Validation Accuracy**: ~96% with MediaPipe, ~98% with YOLOv8.
* High precision across sports and gym actions, including baseball, tennis, and weightlifting.

## Setup & Usage
1. Download the Penn Action Dataset.
2. Run `yolo_extractor.py` or `mediapipe_extractor.py` to generate `.npy` features.
3. Run `train.py` to train the model.
4. Use `visualize.py` to view results.
