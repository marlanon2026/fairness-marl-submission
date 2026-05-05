from typing import List, Dict, Optional, Tuple

# def compute_L1(agent_task_counts: List[int]) -> float:
#     """
#     Compute L1 workload imbalance.
#     L1 = (max workload - min workload) / total subtasks.
    
#     Args:
#         agent_task_counts (List[int]): Number of subtasks each agent completed.
    
#     Returns:
#         float: L1 in [0,1]
#     """
#     if not agent_task_counts:
#         return 0.0
    
#     max_tasks = max(agent_task_counts)
#     min_tasks = min(agent_task_counts)
#     total_tasks = sum(agent_task_counts)
    
#     if total_tasks == 0:
#         return 0.0  # No work was done, define imbalance as 0
    
#     return (max_tasks - min_tasks) / total_tasks

def compute_L1(agent_task_counts: List[int]) -> float:
    """
    Compute L1 workload imbalance using the Gini index.

    Gini = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean)
    Scaled to [0,1] where 0 = perfect equality, 1 = total inequality

    Args:
        agent_task_counts (List[int]): Number of subtasks each agent completed.

    Returns:
        float: L1 (Gini index) in [0,1]
    """
    if not agent_task_counts:
        return 0.0

    n = len(agent_task_counts)
    total_tasks = sum(agent_task_counts)
    if total_tasks == 0:
        return 0.0

    sorted_counts = sorted(agent_task_counts)
    cumulative_diff = 0.0
    for i, xi in enumerate(sorted_counts):
        for xj in sorted_counts:
            cumulative_diff += abs(xi - xj)

    mean = total_tasks / n
    gini = cumulative_diff / (2 * n**2 * mean)
    return gini


def compute_L2(
    task_skill_pairs: List[Tuple[str, int, float]], 
    all_agents_skills: Dict[int, Dict[str, float]]
) -> float:
    """
    Compute normalized L2 skill misalignment according to FairMARL paper.
    
    L2 = 1 - (sum of actual skills used) / (max possible skill sum)
    
    Where max possible skill sum (S_max) is computed by assigning each task 
    to the agent with the highest skill for that task type.
    
    Special cases:
    - When all agents have equal skills: L2 = 0 (any assignment is optimal)
    - When tasks are assigned to least skilled agents: L2 approaches 1
    
    Args:
        task_skill_pairs: List of (task_type, agent_id, skill_used) for each performed task
        all_agents_skills: Dict mapping agent_id to their skill levels for each task type
    
    Returns:
        float: L2 in [0,1] where 0 = optimal skill usage, 1 = worst skill usage
    """
    if not task_skill_pairs:
        return 1.0  # No tasks performed = worst misalignment
    
    # Calculate actual skill sum
    actual_skill_sum = sum(skill for _, _, skill in task_skill_pairs)
    
    # Calculate S_max: for each task performed, what's the max skill available?
    s_max = 0.0
    task_counts = {}
    
    # Count how many times each task type was performed
    for task_type, _, _ in task_skill_pairs:
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    # For each task type, find the maximum skill among all agents
    for task_type, count in task_counts.items():
        max_skill_for_task = 0.0
        for agent_id, skills in all_agents_skills.items():
            skill = skills.get(task_type, 0.0)
            max_skill_for_task = max(max_skill_for_task, skill)
        
        # Add max_skill * count to S_max
        s_max += max_skill_for_task * count
    
    if s_max == 0:
        return 1.0
    
    return 1.0 - (actual_skill_sum / s_max)


def compute_L2_legacy(
    agent_task_skill_sum: List[float],
    agent_task_counts: List[int],
    max_skill_per_task: Optional[List[float]] = None
) -> float:
    """
    Legacy L2 computation - kept for backward compatibility.
    
    This is the old implementation that doesn't match the FairMARL paper.
    Use compute_L2() for the correct implementation.
    """
    total_skill_used = sum(agent_task_skill_sum)
    
    if max_skill_per_task is None or len(max_skill_per_task) == 0:
        # No skill info: treat as worst misalignment
        return 1.0

    max_possible_skill = sum(max_skill_per_task)
    
    if max_possible_skill == 0:
        return 1.0  # Avoid div by zero: worst-case misalignment
    
    return 1.0 - (total_skill_used / max_possible_skill)


def compute_L3(
    L1: float,
    L2: float,
    alpha: float = 0.5
) -> float:
    """
    Compute L3 composite disparity.
    
    L3 = alpha * L1 + (1 - alpha) * L2
    
    Args:
        L1 (float): Workload imbalance [0,1]
        L2 (float): Skill misalignment [0,1]
        alpha (float): Weighting factor [0,1]
    
    Returns:
        float: L3 in [0,1]
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    
    return alpha * L1 + (1.0 - alpha) * L2