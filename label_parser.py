import os
import json
import scipy.io as sio

def create_label_json(mat_folder, output_path):
    labels = {}
    for fname in os.listdir(mat_folder):
        if fname.endswith('.mat'):
            path = os.path.join(mat_folder, fname)
            mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
            video_id = fname.replace('.mat', '')

            annotation = mat.get("annotation", None)
            if annotation is None:
                print(f"Skipping {fname}: 'annotation' key not found.")
                continue

            try:
                action = annotation.action
                labels[video_id] = action
            except Exception as e:
                print(f"Failed to parse {fname}: {e}")
                continue

    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

create_label_json("Penn_Action/labels", "Penn_Action/labels.json")