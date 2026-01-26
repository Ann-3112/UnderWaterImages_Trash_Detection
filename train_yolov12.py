import os
import argparse
from ultralytics import YOLO


def train_yolov12(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    model_name="yolov12s.pt",
    project="runs/detect",
    name="yolov12_trash_main",
):
    """
    Train YOLOv12 model on custom datasets.


    """

    print(f"\n🚀 Initializing Training for Model: {model_name}")

    if not os.path.exists(data):
        print(f"❌ Dataset config '{data}' not found.")
        return

    try:
        model = YOLO(model_name)

        print(f"📈 Starting training for {epochs} epochs...\n")

        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            plots=True,
            save=True,
        )

        # print("\n✅ Training completed successfully!!!")
        print("\n✅ Training completed successfully!!!")
        print(f"📄 Results saved to: {project}/{name}.")
        print(f"🏆 Best model: {project}/{name}/weights/best.pt")

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Training failed: {e}")
        print("💡 Make sure ultralytics is installed:")
        print("   pip install ultralytics")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12 model ")

    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov12s.pt")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="yolov12_trash_main")

    args = parser.parse_args()

    train_yolov12(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        project=args.project,
        name=args.name,
    )
