#!/usr/bin/env python3
"""
Quick launcher for the main dashboard
Run this file to start the RL-based parking pricing dashboard
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard_file = Path(__file__).parent / "main_dashboard.py"
    
    print("\n" + "="*80)
    print(" 🅿  DYNAMIC PARKING PRICING - MAIN DASHBOARD")
    print("="*80)
    print("\nLaunching dashboard...")
    print("Make sure you have pygame and matplotlib installed!")
    print("\nTo install dependencies:")
    print("  pip install pygame matplotlib numpy torch")
    print("\n" + "="*80 + "\n")
    
    try:
        subprocess.run([sys.executable, str(dashboard_file)], check=True)
    except FileNotFoundError:
        print(f"Error: Dashboard file not found at {dashboard_file}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDashboard closed by user")
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        sys.exit(1)
