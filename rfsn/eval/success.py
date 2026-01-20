"""
Evaluation Criteria (No Cheating)
=================================
Explicit success conditions.
"""


def is_success(obs: dict, duration: int) -> bool:
    """
    Determine if episode is successful.
    
    All conditions must be met. If any fail → episode failed.
    
    Args:
        obs: Final observation
        duration: Episode duration in steps
        
    Returns:
        True if episode is successful
    """
    return (
        obs.get("object_lifted", False)
        and obs.get("grasp_stable", False)
        and duration >= 50
        and not obs.get("self_collision", False)
    )
