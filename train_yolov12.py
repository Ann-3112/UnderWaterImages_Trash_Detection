import os
import argparse
from ultralytics import YOLO

def train_yolov12(data='data.yaml', epochs=50, imgsz=640, batch=8, model_name='yolov12s.pt', project='runs/detect', name='yolov12_trash_main'):
    """
    Trains a YOLOv12 model using Ultralytics.
    Ensure that you have the latest version of ultralytics installed or the custom YOLOv12 weights available.
    """
    print(f"🚀 Initializing Training for Model: {model_name}")

    # Verify model existence
    if not os.path.exists(model_name):
        print(f"⚠️ Model file '{model_name}' not found locally. Attempting to download via Ultralytics...")
    
    try:
        # Load the model
        model = YOLO(model_name)
        
        print(f"📈 Starting training for {epochs} epochs...")
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            plots=True,
            save=True
        )
        print("✅ Training completed successfully!")
        print(f"📄 Results saved to {project}/{name}")
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Training failed: {e}")
        print("💡 Suggestion: Ensure 'ultralytics' is up to date (pip install -U ultralytics).")
        print("   If using a custom YOLOv12 model, ensure the .pt file is in the correct path completely.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv12 specific model')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--model', type=str, default='yolov12s.pt', help='Model file (e.g., yolov12n.pt, yolov12s.pt)')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project name')
    parser.add_argument('--name', type=str, default='yolov12_trash_main', help='Run name')

    args = parser.parse_args()
    
    train_yolov12(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
            project='runs/detect',
            name='yolov12_trash_main',
    )
