import os
import json
import shutil
import random

# === CONFIGURATION ===
input_json_path = "C:/Users/NECTEC/Documents/Mask RCNN/pipe_4c/annotations/instances_default.json"
output_val_json_path = "C:/Users/NECTEC/Documents/Mask RCNN/pipe_4c/annotations/val_coco.json"
output_train_json_path = input_json_path  # Overwrite original
images_dir = "C:/Users/NECTEC/Documents/Mask RCNN/pipe_4c/images"   # folder where all original images are
val_images_dir = "C:/Users/NECTEC/Documents/Mask RCNN/pipe_4c/images/val"  # folder to store val images
val_ratio = 0.2
random_seed = 42

# === SETUP ===
os.makedirs(val_images_dir, exist_ok=True)

# === LOAD COCO DATA ===
with open(input_json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

# === SPLIT IMAGES ===
random.seed(random_seed)
random.shuffle(images)

val_count = int(len(images) * val_ratio)
val_images = images[:val_count]
val_image_ids = set(img['id'] for img in val_images)

# === FILTER VAL ANNOTATIONS ===
val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

# === MOVE IMAGES TO VAL FOLDER ===
for img in val_images:
    file_name = img['file_name']
    src_path = os.path.join(images_dir, file_name)
    dst_path = os.path.join(val_images_dir, file_name)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Warning: {file_name} not found in {images_dir}")

# === SAVE VAL JSON ===
val_json = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

with open(output_val_json_path, 'w') as f:
    json.dump(val_json, f, indent=4)

# === REMOVE VAL FROM ORIGINAL AND SAVE UPDATED TRAIN JSON ===
train_images = [img for img in images if img['id'] not in val_image_ids]
train_annotations = [ann for ann in annotations if ann['image_id'] not in val_image_ids]

train_json = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}

with open(output_train_json_path, 'w') as f:
    json.dump(train_json, f, indent=4)

print(f"Moved {len(val_images)} images to {val_images_dir}")
print(f"Saved val annotations to {output_val_json_path}")
print(f"Updated training annotations saved to {output_train_json_path} (excluding val data)")
