from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- MODEL LOADING ----------------
models = {}

# YOLOv8
if os.path.exists("best.pt"):
    print("✅ Loading Custom YOLOv8 Model")
    models["yolov8"] = YOLO("best.pt")
else:
    print("⚠️ Loading fallback YOLOv8 model")
    models["yolov8"] = YOLO("yolov8n.pt")

# YOLOv12
try:
    if os.path.exists("best_yolov12.pt"):
        print("🚀 Loading Custom YOLOv12 Model")
        models["yolov12"] = YOLO("best_yolov12.pt")
    else:
        print("⚠️ Loading fallback YOLOv12 model")
        models["yolov12"] = YOLO("yolov12s.pt")
except Exception as e:
    print("YOLOv12 failed:", e)
    models["yolov12"] = None


# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

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
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    selected_model = request.form.get("model", "yolov8")
    if selected_model not in models:
        return "Invalid model selected", 400

    model = models[selected_model]

    # ---------- SAVE IMAGE ----------
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # ---------- CLAHE ----------
    enhanced_filename = "enhanced_" + filename
    enhanced_path = os.path.join(OUTPUT_FOLDER, enhanced_filename)

    enhance_image(input_path, enhanced_path)

    # ---------- DETECTION ----------
    # YOLOv12 needs lower confidence
    conf_val = 0.25 if selected_model == "yolov8" else 0.1

    # Predict on original image (more stable)
    results = model.predict(input_path, conf=conf_val, iou=0.45)

    # ---------- SAVE RESULT ----------
    detected_filename = "detected_" + filename
    detected_path = os.path.join(OUTPUT_FOLDER, detected_filename)

    res_plotted = results[0].plot()
    cv2.imwrite(detected_path, res_plotted)

    # ---------- EXTRACT DATA -----------
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

    print(f"Detection Complete with {selected_model}: {class_counts}")

    # ---------- FIX PATHS ------------
    web_input_path = "/" + input_path.replace("\\", "/")
    web_enhanced_path = "/" + enhanced_path.replace("\\", "/")
    web_output_path = "/" + detected_path.replace("\\", "/")

    return render_template(
        "index.html",
        input_img=web_input_path,
        enhanced_img=web_enhanced_path,
        output_img=web_output_path,
        results=detections,
        # counts=class_counts
        
        counts=class_counts
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
