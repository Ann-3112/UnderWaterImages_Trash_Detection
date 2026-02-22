import os
import shutil
import pandas as pd
from ultralytics import YOLO

def run_advanced_ofat_yolov11(
    data_yaml="data.yaml",
    base_model="yolov11s.pt",
    tuning_epochs=10,       # Epochs per OFAT test (enough to see a trend)
    final_epochs=50,        # Epochs for the final maximum-confidence training
    project="runs/ofat_yolov11"
):
    print("🚀 Starting Advanced OFAT Tuning for YOLOv11...")
    
    # 1. Define the parameters to test ONE at a time
    ofat_stages = {
        "lr0": [0.01, 0.001, 0.0001],               # Stage 1: Learning Rate
        "optimizer": ["auto", "SGD", "AdamW"],      # Stage 2: Optimizer
        "batch": [4, 8, 16]                         # Stage 3: Batch Size
    }
    
    # Starting base parameters
    best_params = {
        "lr0": 0.01,
        "optimizer": "auto",
        "batch": 8,
        "imgsz": 640
    }
    
    tuning_results = []
    
    # --- PHASE 1: SEQUENTIAL OFAT TUNING ---
    for param_name, values in ofat_stages.items():
        print(f"\n" + "="*50)
        print(f"🔍 TUNING STAGE: {param_name.upper()}")
        print(f"="*50)
        
        stage_best_val = None
        stage_best_map = -1.0
        
        for val in values:
            run_name = f"tune_{param_name}_{val}"
            print(f"\n🧪 Testing {param_name} = {val}...")
            
            # Apply current bests, override with the test value
            current_args = best_params.copy()
            current_args[param_name] = val
            
            try:
                model = YOLO(base_model)
                results = model.train(
                    data=data_yaml,
                    epochs=tuning_epochs,
                    imgsz=current_args["imgsz"],
                    batch=current_args["batch"],
                    lr0=current_args["lr0"],
                    optimizer=current_args["optimizer"],
                    project=project,
                    name=run_name,
                    plots=False, # Save time during tuning
                    save=False   # Don't save weights for intermediate tuning
                )
                
                # Use mAP50-95 as the ultimate indicator of high-confidence bounding boxes
                current_map = results.results_dict.get('metrics/mAP50-95(B)', 0)
                
                tuning_results.append({
                    "Stage": param_name,
                    "Tested_Value": val,
                    "mAP50-95": current_map
                })
                
                print(f"📊 Result for {param_name}={val} -> mAP50-95: {current_map:.4f}")
                
                if current_map > stage_best_map:
                    stage_best_map = current_map
                    stage_best_val = val
                    
            except Exception as e:
                print(f"❌ Run failed for {param_name}={val}: {e}")
                
        # Lock in the best parameter for the next stage
        print(f"\n🏆 Best {param_name} found: {stage_best_val} (mAP: {stage_best_map:.4f})")
        best_params[param_name] = stage_best_val

    # Save Tuning Summary
    df = pd.DataFrame(tuning_results)
    os.makedirs(project, exist_ok=True)
    df.to_csv(os.path.join(project, "tuning_summary.csv"), index=False)
    
    print("\n" + "="*50)
    print("✅ OFAT TUNING COMPLETE. BEST PARAMETERS FOUND:")
    for k, v in best_params.items():
        print(f"   - {k}: {v}")
    print("="*50)

    # --- PHASE 2: FINAL DEEP TRAINING FOR MAXIMUM CONFIDENCE ---
    print(f"\n🔥 Starting FINAL deep training for {final_epochs} epochs to maximize confidence...")
    final_run_name = "yolov11_MAX_CONFIDENCE"
    
    final_model = YOLO(base_model)
    final_model.train(
        data=data_yaml,
        epochs=final_epochs,
        imgsz=best_params["imgsz"],
        batch=best_params["batch"],
        lr0=best_params["lr0"],
        optimizer=best_params["optimizer"],
        project=project,
        name=final_run_name,
        plots=True,
        save=True,
        # Adding augmentations to boost confidence further
        mosaic=1.0, 
        mixup=0.1
    )
    
    # --- PHASE 3: EXPORT FOR FLASK APP ---
    best_weights_path = os.path.join(project, final_run_name, "weights", "best.pt")
    target_app_path = "best_yolov11.pt"
    
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, target_app_path)
        print(f"\n🎉 SUCCESS! The highly optimized model was saved as '{target_app_path}'.")
        print("Your Flask app will automatically use this model for maximum confidence!")
    else:
        print("⚠️ Final weights not found. Check the runs directory.")

if __name__ == "__main__":
    # Ensure ultralytics is ready
    run_advanced_ofat_yolov11()