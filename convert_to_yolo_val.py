import json
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# Check your file structure to ensure these paths match exactly
SOURCE_JSON = "models/Dataset/dataset/material_version/instances_val_trashcan.json"
SOURCE_IMAGES_DIR = "models/Dataset/dataset/material_version/val"

# This is where the new YOLO data will be created
DEST_BASE = "models/Dataset/dataset/material_version"
DEST_IMAGES = os.path.join(DEST_BASE, "val")
DEST_LABELS = os.path.join(DEST_BASE, "val")

def convert():
    if not os.path.exists(SOURCE_JSON):
        print(f"❌ Error: Could not find {SOURCE_JSON}")
        return

    print(f"Loading {SOURCE_JSON}...")
    with open(SOURCE_JSON) as f:
        data = json.load(f)

    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_LABELS, exist_ok=True)

    # Map image IDs to file names
    images_info = {img['id']: img for img in data['images']}

    print("Converting annotations...")
    for ann in tqdm(data['annotations']):
        img_id = ann['image_id']
        img_data = images_info[img_id]
        file_name = img_data['file_name']

        # Calculate YOLO format (center_x, center_y, width, height) normalized 0-1
        img_w = img_data['width']
        img_h = img_data['height']
        x, y, w, h = ann['bbox']

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        # Adjust category ID (YOLO needs 0-15, JSON might be 1-16)
        cat_id = ann['category_id'] - 1

        # Write Label File
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(DEST_LABELS, txt_name), 'a') as f:
            f.write(f"{cat_id} {x_center} {y_center} {width} {height}\n")

        # Copy Image File
        src_img = os.path.join(SOURCE_IMAGES_DIR, file_name)
        dst_img = os.path.join(DEST_IMAGES, file_name)

        # Only copy if we haven't already (to save time)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

    print(f"✅ Conversion Done! Data saved in '{DEST_BASE}'")

if __name__ == "__main__":
    convert()
