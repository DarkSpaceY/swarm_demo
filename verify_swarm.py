import subprocess
import json
import os
import time

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def get_accuracy_from_metrics(run_name):
    metrics_path = f"./runs/{run_name}/metrics.jsonl"
    if not os.path.exists(metrics_path):
        return None
    
    accuracies = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                accuracies.append(data.get("accuracy", 0))
            except:
                continue
    return accuracies[-1] if accuracies else 0

def main():
    # 1. Baseline: 1.5B Single Model
    baseline_run = "baseline_1.5B"
    baseline_cmd = [
        "python3", "evolution.py",
        "--run-name", baseline_run,
        "--model-name", "unsloth/Qwen2.5-1.5B-Instruct",
        "--num-batches", "1",
        "--num-problems", "50",
        "--num-solver-samples", "1",
        "--num-judge-samples", "0",
        "--max-rounds", "1",
        "--agent-count", "1",
        "--no-train-lora",
        "--no-train-judge",
        "--no-train-preference"
    ]
    
    print("=== Step 1: Running Baseline (1.5B Single) ===")
    run_command(baseline_cmd)
    baseline_acc = get_accuracy_from_metrics(baseline_run)
    print(f"Baseline Accuracy: {baseline_acc:.2%}" if baseline_acc is not None else "Baseline failed.")

    # 2. Swarm: 0.5B + 5 Agents (Swarm)
    swarm_run = "swarm_0.5B_5agents"
    swarm_cmd = [
        "python3", "evolution.py",
        "--run-name", swarm_run,
        "--model-name", "unsloth/Qwen2.5-0.5B-Instruct",
        "--num-batches", "1",
        "--num-problems", "50",
        "--num-solver-samples", "2",
        "--num-judge-samples", "6",
        "--max-rounds", "5",
        "--agent-count", "5",
        "--no-train-lora",
        "--no-train-judge",
        "--no-train-preference"
    ]
    
    print("\n=== Step 2: Running Swarm (0.5B Swarm of 5) ===")
    run_command(swarm_cmd)
    swarm_acc = get_accuracy_from_metrics(swarm_run)
    print(f"Swarm Accuracy: {swarm_acc:.2%}" if swarm_acc is not None else "Swarm failed.")

    # 3. Summary
    print("\n=== Summary ===")
    if baseline_acc is not None and swarm_acc is not None:
        diff = swarm_acc - baseline_acc
        print(f"Baseline (1.5B): {baseline_acc:.2%}")
        print(f"Swarm (0.5B x 5): {swarm_acc:.2%}")
        print(f"Difference: {diff:+.2%}")
        if diff > 0:
            print("Conclusion: Swarm (0.5B x 5) OUTPERFORMS 1.5B Single Model!")
        else:
            print("Conclusion: 1.5B Single Model still leads.")
    else:
        print("Comparison incomplete due to errors.")

if __name__ == "__main__":
    main()
