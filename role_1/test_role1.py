"""
Test Suite

Comprehensive tests to validate all ROLE 1 components.
"""

import numpy as np
import sys
from env import ParkingPricingEnv
from metrics import compute_all_metrics, ParkingMetrics
from reward_function import RewardFunction
from state_action_documentation import *


def test_environment_initialization():
    """Test environment can be created and initialized."""
    print("TEST 1: Environment Initialization")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv(capacity=100, max_steps=288)
        print(f"✓ Environment created")
        print(f"  - Capacity: {env.capacity}")
        print(f"  - Max steps: {env.max_steps}")
        print(f"  - State space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_environment_reset():
    """Test environment reset functionality."""
    print("\nTEST 2: Environment Reset")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv()
        obs, info = env.reset()
        
        assert obs.shape == (5,), f"Wrong obs shape: {obs.shape}"
        assert isinstance(info, dict), f"Info should be dict: {type(info)}"
        assert 0.4 <= env.occupancy <= 0.6, "Occupancy not in [0.4, 0.6]"
        assert env.time_step == 0, "Time step should be 0 after reset"
        
        print(f"✓ Reset works correctly")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Initial occupancy: {env.occupancy:.2%}")
        print(f"  - Initial time step: {env.time_step}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_environment_step():
    """Test environment step functionality."""
    print("\nTEST 3: Environment Step")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv()
        obs, _ = env.reset()
        
        # Take a step with price $10
        action = np.array([10.0])
        obs_new, reward, terminated, truncated, info = env.step(action)
        
        assert obs_new.shape == (5,), f"Wrong obs shape: {obs_new.shape}"
        assert isinstance(reward, (float, np.floating)), f"Reward should be float: {type(reward)}"
        assert isinstance(terminated, (bool, np.bool_)), f"Terminated should be bool"
        assert "revenue" in info, "Missing revenue in info"
        assert "occupancy" in info, "Missing occupancy in info"
        assert "price" in info, "Missing price in info"
        
        assert env.time_step == 1, "Time step not incremented"
        
        print(f"✓ Step works correctly")
        print(f"  - New state shape: {obs_new.shape}")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Revenue: ${info['revenue']:.2f}")
        print(f"  - New occupancy: {info['occupancy']:.2%}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_episode_execution():
    """Test full episode execution."""
    print("\nTEST 4: Full Episode Execution")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv(max_steps=288)
        obs, _ = env.reset()
        
        step_count = 0
        for step in range(288):
            action = np.array([10.0])  # Fixed $10 price
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated:
                break
        
        assert step_count == 288, f"Episode length wrong: {step_count}"
        assert env.time_step == 288, "Time step not updated correctly"
        
        print(f"✓ Episode completed successfully")
        print(f"  - Steps executed: {step_count}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_metrics_computation():
    """Test metrics computation."""
    print("\nTEST 5: Metrics Computation")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv(max_steps=288)
        obs, _ = env.reset()
        
        # Run episode
        for step in range(288):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        # Compute metrics
        metrics = env.get_episode_metrics()
        
        assert isinstance(metrics, dict), f"Metrics should be dict: {type(metrics)}"
        assert "total_revenue" in metrics, "Missing total_revenue"
        assert "avg_occupancy" in metrics, "Missing avg_occupancy"
        assert "occupancy_std" in metrics, "Missing occupancy_std"
        assert "price_volatility" in metrics, "Missing price_volatility"
        
        # Check value ranges
        assert metrics["total_revenue"] > 0, "Revenue should be positive"
        assert 0 <= metrics["avg_occupancy"] <= 1, "Occupancy not in [0,1]"
        assert metrics["occupancy_std"] >= 0, "Std should be non-negative"
        assert metrics["price_volatility"] >= 0, "Volatility should be non-negative"
        
        print(f"✓ Metrics computed successfully")
        print(f"  - Total revenue: ${metrics['total_revenue']:.2f}")
        print(f"  - Avg occupancy: {metrics['avg_occupancy']:.1%}")
        print(f"  - Occupancy std: {metrics['occupancy_std']:.1%}")
        print(f"  - Price volatility: ${metrics['price_volatility']:.2f}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_reward_function():
    """Test reward function implementation."""
    print("\nTEST 6: Reward Function")
    print("-" * 60)
    
    try:
        reward_fn = RewardFunction()
        
        # Test components
        r_rev = reward_fn.compute_revenue_reward(0.8, 10.0)
        r_occ = reward_fn.compute_occupancy_reward(0.8)
        r_vol = reward_fn.compute_volatility_reward(10.0, 10.0)
        
        assert isinstance(r_rev, (float, np.floating)), "Revenue should be float"
        assert isinstance(r_occ, (float, np.floating)), "Occupancy should be float"
        assert isinstance(r_vol, (float, np.floating)), "Volatility should be float"
        
        # Test total reward
        total_reward, components = reward_fn.compute_total_reward(0.8, 10.0, 10.0)
        
        assert isinstance(total_reward, float), "Total reward should be float"
        assert isinstance(components, dict), "Components should be dict"
        assert "revenue" in components, "Missing revenue component"
        assert "occupancy" in components, "Missing occupancy component"
        assert "volatility" in components, "Missing volatility component"
        
        # Test batch computation
        occupancies = np.array([0.7, 0.8, 0.9])
        prices = np.array([8.0, 10.0, 12.0])
        prices_prev = np.array([8.0, 10.0, 12.0])
        
        batch_rewards = reward_fn.batch_compute_reward(occupancies, prices, prices_prev)
        
        assert batch_rewards.shape == (3,), f"Wrong batch shape: {batch_rewards.shape}"
        assert all(isinstance(r, (float, np.floating)) for r in batch_rewards), "Batch should have floats"
        
        print(f"✓ Reward function works correctly")
        print(f"  - Revenue reward: {r_rev:.4f}")
        print(f"  - Occupancy reward: {r_occ:.4f}")
        print(f"  - Volatility reward: {r_vol:.4f}")
        print(f"  - Total reward: {total_reward:.4f}")
        print(f"  - Batch rewards: {batch_rewards}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_state_space_bounds():
    """Test state space stays within bounds."""
    print("\nTEST 7: State Space Bounds")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv()
        obs, _ = env.reset()
        
        for step in range(288):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check bounds
            assert obs[0] >= 0 and obs[0] <= 1, f"Occupancy out of bounds: {obs[0]}"
            assert obs[1] >= 0 and obs[1] <= 1, f"Time out of bounds: {obs[1]}"
            assert obs[2] >= 0 and obs[2] <= 1, f"Demand out of bounds: {obs[2]}"
            
            if terminated:
                break
        
        print(f"✓ State space bounds maintained throughout episode")
        print(f"  - Final occupancy: {obs[0]:.2%}")
        print(f"  - Final time: {obs[1]:.2f}")
        print(f"  - Final demand: {obs[2]:.2%}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_action_space_bounds():
    """Test action space enforcement."""
    print("\nTEST 8: Action Space Bounds")
    print("-" * 60)
    
    try:
        env = ParkingPricingEnv()
        obs, _ = env.reset()
        
        # Test actions outside bounds (should be clipped)
        test_prices = [-5.0, 0.5, 10.0, 20.0, 25.0]
        
        for test_price in test_prices:
            action = np.array([test_price])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Price should be clipped to [0.5, 20.0]
            actual_price = info["price"]
            expected_price = np.clip(test_price, 0.5, 20.0)
            
            assert abs(actual_price - expected_price) < 0.01, \
                f"Price not clipped correctly: {actual_price} vs {expected_price}"
        
        print(f"✓ Action space bounds enforced correctly")
        print(f"  - Test prices: {test_prices}")
        print(f"  - All clipped to [0.5, 20.0]")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("ROLE 1: COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_environment_initialization,
        test_environment_reset,
        test_environment_step,
        test_episode_execution,
        test_metrics_computation,
        test_reward_function,
        test_state_space_bounds,
        test_action_space_bounds,
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_func.__name__}")
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - ROLE 1 COMPLETE!")
        return 0
    else:
        print("\n✗ Some tests failed - review above")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
