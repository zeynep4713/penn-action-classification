import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
import random
from collections import defaultdict

class PoseDatasetFromNpy(Dataset):
    def __init__(self, npy_folder, labels_json, split="train"):
        with open(labels_json, 'r') as f:
            all_labels = json.load(f)

        # Group video_ids by class
        class_to_videos = defaultdict(list)
        for vid, label in all_labels.items():
            class_to_videos[label].append(vid)

        train_videos, val_videos = [], []
        for vids in class_to_videos.values():
            random.shuffle(vids)
            split_idx = int(0.8 * len(vids))
            train_videos.extend(vids[:split_idx])
            val_videos.extend(vids[split_idx:])

        selected = train_videos if split == "train" else val_videos
        self.labels_dict = {vid: all_labels[vid] for vid in selected}
        self.samples = []
        self.labels = []
        self.classes = sorted(set(all_labels.values()))
        self.encoder = LabelEncoder().fit(self.classes)

        for video_id, label in self.labels_dict.items():
            npy_path = os.path.join(npy_folder, video_id + ".npy")
            if not os.path.exists(npy_path):
                continue
            keypoints = np.load(npy_path)
            self.samples.append(keypoints)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (T, 17, 3)
        x = x.view(x.shape[0], -1)  # flatten to (T, 51)
        length = x.shape[0]
        y = torch.tensor(self.encoder.transform([self.labels[idx]])[0], dtype=torch.long)
        return x, length, y
