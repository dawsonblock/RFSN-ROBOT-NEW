"""
Deterministic Replay Harness
============================
If replay fails, your system is nondeterministic.
"""

import json
from typing import Any, List


def replay(log_path: str, env: Any) -> bool:
    """
    Replay a logged episode and verify determinism.
    
    Args:
        log_path: Path to episode log (JSONL)
        env: Environment to replay in
        
    Returns:
        True if replay matches exactly
        
    Raises:
        AssertionError if nondeterminism detected
    """
    with open(log_path, "r") as f:
        log = [json.loads(line) for line in f]
    
    env.reset(randomize=False)
    
    for entry in log:
        if "action" in entry:
            env.step(entry["action"])
            
        if "obs" in entry:
            current_obs = env.observe()
            # Compare key observation values
            for key in ["q", "dq", "ee_pos"]:
                if key in entry["obs"] and key in current_obs:
                    expected = entry["obs"][key]
                    actual = current_obs[key]
                    assert expected == actual, f"Nondeterminism at step {entry['step']}: {key}"
    
    return True


def load_log(log_path: str) -> List[dict]:
    """Load an episode log from file."""
    with open(log_path, "r") as f:
        return [json.loads(line) for line in f]
