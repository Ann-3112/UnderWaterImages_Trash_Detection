from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -- MODEL LOADING LOGIC --
# UPDATED: Look for the file you downloaded from Colab in the root folder
TRAINED_MODEL_PATH = "best.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt"

if os.path.exists(TRAINED_MODEL_PATH):
    print(f"✅ Loading Custom Trained Model: {TRAINED_MODEL_PATH}")
    model = YOLO(TRAINED_MODEL_PATH)
else:
    print(f"⚠️ Custom model '{TRAINED_MODEL_PATH}' not found!")
    print(f"ℹ️ Loading Generic Model: {FALLBACK_MODEL_PATH} (Detection will not be accurate)")
    model = YOLO(FALLBACK_MODEL_PATH)

def enhance_image(image_path, output_path):
    """
    Applies CLAHE enhancement to the uploaded image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, enhanced)
    return True

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return "No file uploaded", 400
        
    file = request.files["image"]
    if file.filename == '':
        return "No file selected", 400

    # --- 1. SAVE UPLOADED IMAGE ---
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # --- 2. ENHANCE IMAGE ---
    enhanced_filename = "enhanced_" + filename
    enhanced_path = os.path.join(OUTPUT_FOLDER, enhanced_filename)
    
    success = enhance_image(input_path, enhanced_path)
    if not success:
        return "Error processing image", 500

    # --- 3. RUN YOLO DETECTION ---
    # Run on enhanced image
    results = model.predict(enhanced_path, conf=0.25, iou=0.45)

    # --- 4. SAVE RESULT ---
    detected_filename = "detected_" + filename
    detected_path = os.path.join(OUTPUT_FOLDER, detected_filename)
    
    # Save the plotted image
    res_plotted = results[0].plot()
    cv2.imwrite(detected_path, res_plotted)

    # --- 5. EXTRACT DATA FOR UI ---
    detections = []
    class_counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        
        detections.append({
            "class": cls_name,
            "conf": f"{round(conf * 100, 1)}%"
        })

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f"Detection Complete: {class_counts}")

    # --- 6. FIX PATHS FOR HTML ---
    # Windows uses backslashes (\), but browsers need forward slashes (/)
    # We also add a leading slash so Flask knows it's from the root
    web_input_path = "/" + input_path.replace("\\", "/")
    web_enhanced_path = "/" + enhanced_path.replace("\\", "/")
    web_output_path = "/" + detected_path.replace("\\", "/")

    return render_template(
        "index.html",
        input_img=web_input_path,
        enhanced_img=web_enhanced_path,
        output_img=web_output_path,
        results=detections,
        counts=class_counts
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)