#!/usr/bin/env python
"""
Quick script to verify fairness components are working
File Path: ./verify_fairness.py

Usage:
  cd <repo-root>
  python verify_fairness.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test 1: Verify imports
print("=== TEST 1: Checking imports ===")
try:
    from utils.adaptive_lambda_scheduler import AdaptiveLambdaScheduler
    from utils.reward_normalizer import AdaptiveRewardNormalizer
    from utils.fairness_reward_handler import FairnessRewardHandler
    print("✓ All fairness modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Test Lambda Scheduler
print("\n=== TEST 2: Testing Lambda Scheduler ===")
scheduler = AdaptiveLambdaScheduler(
    initial_lambda=0.5,
    target_lambda=10.0,
    warmup_episodes=1000,
    schedule_type='cosine'
)

test_episodes = [0, 50, 100, 200, 400]
for ep in test_episodes:
    scheduler.current_episode = ep
    lambda_val = scheduler.get_lambda()
    progress = scheduler.get_progress()
    print(f"Episode {ep:4d}: λ={lambda_val:6.2f}, Progress={progress:5.1f}%")

# Test 3: Test Reward Normalizer
print("\n=== TEST 3: Testing Reward Normalizer ===")
normalizer = AdaptiveRewardNormalizer(window_size=100)

# Simulate some rewards
import numpy as np
for i in range(150):
    base = np.random.uniform(0.1, 1.0)
    penalty = np.random.uniform(0, 0.5)
    normalized = normalizer.normalize_reward(base, penalty, 10.0)
    
    if i % 50 == 0:
        stats = normalizer.get_stats()
        if stats:
            print(f"Step {i:3d}: base={base:.3f}, penalty={penalty:.3f}, "
                  f"normalized={normalized:.3f}, avg_final={stats['avg_final']:.3f}")

# Test 4: Check if fairness is enabled in environment
print("\n=== TEST 4: Checking Environment Config ===")
import json
config_path = "./environments/env_generator/examples/multiagent_giveshock_specforced_coop.json"
with open(config_path) as f:
    config = json.load(f)
    
print(f"use_fairness: {config['config'].get('use_fairness', False)}")
print(f"lambda_fairness: {config['config'].get('lambda_fairness', 0)}")
print(f"initial_lambda: {config['config'].get('initial_lambda', 0)}")
print(f"warmup_episodes: {config['config'].get('warmup_episodes', 0)}")

print("\n✅ All tests completed!")