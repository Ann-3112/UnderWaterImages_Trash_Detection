import os
import argparse
from ultralytics import YOLO

def train_yolov12(data='data.yaml', epochs=50, imgsz=640, batch=8, model_name='yolov12s.pt', project='runs/detect', name='yolov12_trash_main'):
    print(f"🚀 Initializing Training for Model: {model_name}")

    try:
        # Load the model
        model = YOLO(model_name)
        
        print(f"📈 Starting training for {epochs} epochs...")
        results = model.train(
            data=data,       # Path to your data.yaml
            epochs=epochs,   # Number of passes through the data
            imgsz=imgsz,     # Image resolution
            batch=batch,     # Images per step
            project=project, # Folder to save results
            name=name,       # Name of this specific run
            plots=True,
            save=True,
            
            # --- 🌊 UNDERWATER TUNING FIXES ---
            lr0=0.001,       # Lower learning rate for YOLOv12 stability
            cls=2.0,         # Higher penalty for wrong classification
            mosaic=0.5,      # Reduced mosaic to help with underwater haze
            copy_paste=0.3   # Helps with rare trash classes
        )
        print("✅ Training completed successfully!")
        return results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv12 specific model')
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model', type=str, default='yolov12s.pt')

    args = parser.parse_args()
    
    # Passing the args directly into the function
    train_yolov12(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model
    )