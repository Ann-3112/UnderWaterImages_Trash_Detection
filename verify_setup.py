import sys
try:
    import ultralytics
    from ultralytics import YOLO
    
    # Try initializing a YOLOv12 model
    try:
        print("Attempting to load YOLOv12s...")
        model = YOLO('yolov12s.pt')
        print("✅ YOLOv12 model loaded successfully!")
    except Exception as e_yolo12:
        print(f"⚠️ Could not load YOLOv12: {e_yolo12}")
        print("Attemping to load YOLOv8n as fallback validation...")
        try:
            model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loaded successfully (Ultralytics is working, just missing YOLOv12 weights/support).")
        except Exception as e_yolo8:
            print(f"❌ Could not load YOLOv8 either: {e_yolo8}")

except ImportError:
    print("❌ Ultralytics not installed.")
except Exception as e:
    print(f"❌ Error loading YOLOv12: {e}")
