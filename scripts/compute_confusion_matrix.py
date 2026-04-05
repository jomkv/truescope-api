import json

def compute_detailed_metrics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 2x2 Matrix: [[TN, FP], [FN, TP]]
    # Ground Truth: FALSE, TRUE
    # Predicted: FALSE, TRUE
    
    matrix = {
        "TP": 0, # True says True
        "TN": 0, # False says False
        "FP": 0, # False says True
        "FN": 0  # True says False
    }
    
    for r in results:
        gt = r['ground_truth'].upper()
        pred = r['predicted_label'].upper()
        
        if gt == "TRUE" and pred == "TRUE":
            matrix["TP"] += 1
        elif gt == "FALSE" and pred == "FALSE":
            matrix["TN"] += 1
        elif gt == "FALSE" and pred == "TRUE":
            matrix["FP"] += 1
        elif gt == "TRUE" and pred == "FALSE":
            matrix["FN"] += 1
            
    return matrix

if __name__ == "__main__":
    path = "tests/test_100_claims/test_accuracy_results_20260320_152449.json"
    m = compute_detailed_metrics(path)
    print(json.dumps(m, indent=2))
