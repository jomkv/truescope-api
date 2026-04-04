import json

def extract_errors(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    errors = [r for r in results if not r.get('is_correct', True)]
    
    fp = [e for e in errors if e['ground_truth'].upper() == "FALSE"]
    fn = [e for e in errors if e['ground_truth'].upper() == "TRUE"]
    
    return fp, fn

if __name__ == "__main__":
    path = "tests/test_100_claims/test_accuracy_results_20260320_152449.json"
    fp, fn = extract_errors(path)
    
    print("### FALSE POSITIVES (9)")
    for i, e in enumerate(fp):
        print(f"{i+1}. Claim: {e['claim']}\n   Ground Truth: {e['ground_truth']} | Predicted: {e['predicted_label']} | Score: {e['system_score']}\n")
        
    print("\n### FALSE NEGATIVES (9)")
    for i, e in enumerate(fn):
        print(f"{i+1}. Claim: {e['claim']}\n   Ground Truth: {e['ground_truth']} | Predicted: {e['predicted_label']} | Score: {e['system_score']}\n")
