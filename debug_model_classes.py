from ultralytics import YOLO

# Load the model we put in place
model_path = "best_yolov12.pt"
print(f"Loading model from: {model_path}")

try:
    model = YOLO(model_path)
    
    # Print class names
    print("Model Classes:")
    print(model.names)
    
    # Check specifically for 'snowboard' or 'person'
    if 0 in model.names and model.names[0] == 'person':
        print("\n⚠️ WARNING: This model appears to be trained on COCO (Person, Bicycle, etc.)")
    elif 0 in model.names and model.names[0] == 'rov':
         print("\n✅ SUCCESS: This model appears to be trained on the Custom Trash Dataset")
    else:
        print(f"\nℹ️ First class is: {model.names.get(0, 'Unknown')}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
