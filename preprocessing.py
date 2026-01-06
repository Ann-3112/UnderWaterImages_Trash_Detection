import cv2
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# This path must point to the images generated in Step 1
INPUT_FOLDER = "yolo_dataset/train/images"

def apply_clahe_enhancement():
    # 1. Verification: Check if Step 1 was actually done
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: The folder '{INPUT_FOLDER}' does not exist.")
        print("   Please make sure Step 1 (convert_to_yolo.py) ran successfully.")
        return

    # 2. Get list of all image files
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(image_files) == 0:
        print("❌ Error: No images found in the folder.")
        return

    print(f"🌊 Found {len(image_files)} images. Starting Underwater Enhancement...")

    # 3. Setup CLAHE algorithm
    # clipLimit=2.5 is usually the best setting for underwater visibility
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # 4. Loop through every image and enhance it
    for filename in tqdm(image_files, desc="Enhancing"):
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        # Read the image
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        try:
            # --- Image Processing Logic ---
            # Convert BGR (Blue Green Red) to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Split into channels: L (Lightness), A (Color), B (Color)
            l, a, b = cv2.split(lab)

            # Apply enhancement ONLY to the Lightness channel
            l_enhanced = clahe.apply(l)

            # Merge channels back together
            merged = cv2.merge((l_enhanced, a, b))

            # Convert back to standard BGR format
            final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            # Overwrite the original image with the enhanced version
            # This ensures the AI trains on the clear images
            cv2.imwrite(img_path, final_img)
            
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    print("\n✅ Preprocessing Complete!")
    print("   All images in 'yolo_dataset/train/images' are now enhanced.")

if __name__ == "__main__":
    apply_clahe_enhancement()