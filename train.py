from ultralytics import YOLO
import os

def start_training():
    # 1. Verification
    if not os.path.exists("data.yaml"):
        print("❌ Error: 'data.yaml' not found. Please complete Step 3.")
        return

    print("🚀 Loading YOLOv8 Model..")
    # We load 'yolov8n.pt' (Nano). It is the fastest model, perfect for laptops.
    # It will download automatically if you don't have it.
    model = YOLO("yolov8n.pt") 

    print("🔥 Starting Training... (This will take time!)")
    
    # 2. Start Training
    # data: Points to your yaml file
    # epochs: How many times to review the data (30 is a good balance for speed/accuracy)
    # imgsz: Standard image size (640x640)
    # batch: How many images to process at once. 
    #        If you get a "Memory Error", change batch to 4 or 2.
    # name: The folder where results will be saved.
    model.train(
        data="data.yaml",
        epochs=30,          
        imgsz=640,
        batch=8,            
        name="trash_training_run",
        plots=True          # Generates graphs showing how the model is learning
    )

    print("\n✅ Training Finished!")
    print("---------------------------------------------------------")
    print("YOUR MODEL IS SAVED AT:")
    print("runs/detect/trash_training_run/weights/best.pt")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    # On Windows, this line is strictly required to prevent crashing
    start_training()