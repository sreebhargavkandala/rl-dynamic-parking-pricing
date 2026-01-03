#!/usr/bin/env python3
"""
QUICK TRAINING LAUNCHER
Simply run this to train the agent with best settings
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    train_script = Path(__file__).parent / "train_optimized.py"
    
    print("\n" + "="*80)
    print("  STARTING OPTIMIZED A2C AGENT TRAINING")
    print("="*80 + "\n")
    
    print("Training settings:")
    print("  • Episodes: 1000")
    print("  • Algorithm: A2C (Actor-Critic)")
    print("  • Hyperparameters: Optimized")
    print("  • Device: GPU (if available) / CPU")
    print("  • Early stopping: Yes (if no improvement for 100 episodes)\n")
    
    try:
        subprocess.run([sys.executable, str(train_script)], check=True)
    except KeyboardInterrupt:
        print("\n⚠️  Training cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
