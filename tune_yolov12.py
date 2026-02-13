from ultralytics import YOLO

# 1. Load your YOLOv12 model (adjust path to your .pt or .yaml)
model = YOLO("yolov12s.pt")

# 2. Run the automated tuner
# This creates a 'runs/detect/tune' folder with the best results
results = model.tune(
    data="your_dataset.yaml", 
    epochs=50,           # Epochs per iteration
    iterations=100,      # Number of tuning iterations
    optimizer="AdamW", 
    plots=False, 
    save=False, 
    val=True
)