"""
Evaluation Harness (Truth-Telling)
==================================
No cheating. Deterministic evaluation.
"""

import csv
from typing import Callable, Any


def run_eval(env: Any, policy: Callable, episodes: int = 100,
             output_path: str = "eval_results.csv"):
    """
    Run deterministic evaluation.
    
    Args:
        env: Environment with reset(), observe(), step(), done() methods
        policy: Policy function that takes obs and returns action
        episodes: Number of episodes to evaluate
        output_path: Path to save results CSV
    """
    rows = []
    
    for ep in range(episodes):
        env.reset(randomize=True)
        done = False
        success = True
        steps = 0

        while not done:
            obs = env.observe()
            action = policy(obs)
            ok, reason = env.step(action)
            steps += 1
            
            if not ok:
                success = False
                break
            done = env.done()

        rows.append({
            "episode": ep,
            "success": int(success),
            "steps": steps
        })

    # Write results
    with open(output_path, "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=["episode", "success", "steps"])
        w.writeheader()
        w.writerows(rows)
    
    # Print summary
    success_rate = sum(r["success"] for r in rows) / len(rows)
    print(f"Evaluation complete: {episodes} episodes, {success_rate:.1%} success")
    
    return rows
