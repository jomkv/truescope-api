import json
import glob
import os
import sys

def main():
    # Find the most recent test results if none provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Search relative to this script's location
        search_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_accuracy_results_*.json")
        files = glob.glob(search_path)
        if not files:
            print(f"No test_accuracy_results JSON found in {os.path.dirname(os.path.abspath(__file__))}.")
            return
        filepath = max(files, key=os.path.getmtime)
    
    print(f"Analyzing {os.path.basename(filepath)}...\n")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
    
    results = data.get("results", [])
    incorrect = [r for r in results if not r.get("is_correct")]
    
    print(f"Overall Accuracy: {data.get('accuracy', 0) * 100:.1f}%")
    print(f"Total Incorrect: {len(incorrect)} / {data.get('total', 0)}\n")
    print("="*80)
    
    for item in incorrect:
        print(f"\n[ID: {item.get('id')}] CLAIM: {item.get('claim')}")
        print(f"Ground Truth: {item.get('ground_truth')} | Predicted: {item.get('predicted_label')} (Score: {item.get('system_score')})")
        print("-" * 40)
        
        evidence_list = item.get("evidence_debug", [])
        if not evidence_list:
            print("  No evidence found for this claim.")
        
        for i, ev in enumerate(evidence_list):
            print(f"  Evidence {i+1} [{ev.get('source')}]:")
            print(f"  Similarity: {ev.get('similarity'):.4f} | NLI: {ev.get('nli_label', 'UNKNOWN').upper()} (Conf: {ev.get('nli_score', 0):.4f})")
            snippet = ev.get('snippet', '').replace('\n', ' ')
            print(f"  Snippet: {snippet}")
            print()
            
        print("="*80)

if __name__ == "__main__":
    main()
