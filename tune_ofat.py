import argparse
import csv
import os
import sys
from ultralytics import YOLO

def run_ofat_tuning(
    model_path='yolov8n.pt',
    data_path='data.yaml',
    base_epochs=10,
    output_file='ofat_results.csv',
    search_space=None
):
    """
    Runs One-Factor-At-A-Time (OFAT) hyperparameter tuning.
    Iterates through each parameter in the search space, modifying it while keeping others at baseline.
    """
    
    print(f"🚀 Starting OFAT Tuning with model: {model_path}")
    print(f"📂 Data: {data_path}")
    print(f"💾 Results will be saved to: {output_file}")

    # Baseline Hyperparameters (defaults closer to standard YOLO settings)
    # These are the "center" points. When we tune 'lr0', we move away from this baseline.
    baseline_params = {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

    # Default Search Space if none provided
    if search_space is None:
        search_space = {
            'lr0': [0.001, 0.01, 0.1],
            'momentum': [0.8, 0.937, 0.98],
            'weight_decay': [0.0001, 0.0005, 0.001],
            'box': [1.0, 7.5, 15.0],
        }

    # Initialize CSV with headers
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Parameter', 'Value', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'Fitness'])

    # 1. Baseline Performance (Optional: Run once with all baselines)
    # We will just proceed to the loops. If the baseline value is in the search list, it gets run.

    for param, values in search_space.items():
        print(f"\n🔍 Tuning Parameter: {param}")
        
        for val in values:
            print(f"   ▶ Testing {param} = {val}")
            
            # Prepare arguments
            # Copy baseline and override the specific parameter being tuned
            current_args = baseline_params.copy()
            current_args[param] = val
            
            run_name = f"tune_{param}_{val}".replace('.', 'p') # Sanitize name
            
            try:
                # Load model fresh for each run to avoid state leakage
                model = YOLO(model_path)
                
                # Train
                # We use a purely unique project dir for tuning to avoid cluttering runs/detect
                results = model.train(
                    data=data_path,
                    epochs=base_epochs,
                    project='runs/tune',
                    name=run_name,
                    exist_ok=True,
                    plots=False, # Save disk space
                    save=False,  # Save disk space, we only care about metrics
                    **current_args
                )
                
                # Extract metrics
                # Ultralytics results.results_dict keys might vary slightly by version, 
                # but usually: metrics/mAP50(B), metrics/mAP50-95(B), fitness
                metrics = results.results_dict
                if metrics:
                    map50 = metrics.get('metrics/mAP50(B)', 0)
                    map5095 = metrics.get('metrics/mAP50-95(B)', 0)
                    precision = metrics.get('metrics/precision(B)', 0)
                    recall = metrics.get('metrics/recall(B)', 0)
                    fitness = metrics.get('fitness', 0)
                else:
                    # Fallback if results_dict is empty (rare)
                    map50 = 0
                    map5095 = 0
                    precision = 0
                    recall = 0
                    fitness = 0

                print(f"      ✅ Result: mAP50={map50:.4f}, Fitness={fitness:.4f}")

                # Log to CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([param, val, map50, map5095, precision, recall, fitness])

            except Exception as e:
                print(f"      ❌ Failed run for {param}={val}: {e}")

    print(f"\n🎉 OFAT Tuning Complete! Check {output_file} for results.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OFAT Hyperparameter Tuning for YOLO')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path (e.g. yolov8n.pt, yolov12s.pt)')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to dataset yaml')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per tuning run')
    parser.add_argument('--output', type=str, default='ofat_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    run_ofat_tuning(
        model_path=args.model,
        data_path=args.data,
        base_epochs=args.epochs,
        output_file=args.output
    )
