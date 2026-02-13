from ultralytics import YOLO

def test_yolov12_loading():
    try:
        print("🚀 Attempting to load YOLOv12 model: yolov12s.pt")
        model = YOLO("yolov12s.pt")
        print("✅ YOLOv12 model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLOv12: {e}")
        return False

if __name__ == "__main__":
    test_yolov12_loading()
