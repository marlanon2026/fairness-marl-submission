import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fairness_metrics_cal import compute_L1

# Test cases
print("Equal workload [3, 3, 3] →", compute_L1([3, 3, 3]))     # Expected: 0.0
print("Slight imbalance [2, 3, 5] →", compute_L1([2, 3, 5]))   # Expected: small value ~0.2
print("Extreme imbalance [0, 0, 9] →", compute_L1([0, 0, 9]))  # Expected: high value close to 1.0
print("No work done [0, 0, 0] →", compute_L1([0, 0, 0]))       # Expected: 0.0
