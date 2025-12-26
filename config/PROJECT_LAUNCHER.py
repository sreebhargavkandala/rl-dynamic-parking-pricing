"""
PARKING LOT PRICING PROJECT - MASTER LAUNCHER
==============================================
Central hub for all simulators and dashboards
Complete project management system
"""

import subprocess
import sys
import os
from pathlib import Path

SIMULATORS = {
    "1": {
        "name": "RL-Integrated City Parking Simulator (RECOMMENDED)",
        "file": "rl_integrated_simulator.py",
        "description": "Real city parking with 60% min occupancy + Trained RL pricing model",
        "features": ["Q-Learning Training", "Day-by-day progression", "Real-time pricing"]
    },
    "2": {
        "name": "Day-by-Day Parking Simulator",
        "file": "day_by_day_simulator.py",
        "description": "Step-by-step daily simulation with previous day comparison",
        "features": ["Manual day progression", "Revenue tracking", "Weather effects"]
    },
    "3": {
        "name": "Realistic Parking Demo",
        "file": "realistic_parking_demo.py",
        "description": "Beautiful 3D-style parking lot with algorithm breakdown",
        "features": ["Realistic graphics", "Price calculation display", "Slow motion mode"]
    },
    "4": {
        "name": "Demo Simulator",
        "file": "demo_simulator.py",
        "description": "Professional demo showing algorithm transparency",
        "features": ["Clear pricing breakdown", "Hover details", "Faculty-ready"]
    },
    "5": {
        "name": "Real-Time Dashboard",
        "file": "dashboard.py",
        "description": "Comprehensive monitoring and analytics dashboard",
        "features": ["Live metrics", "Charts & graphs", "RL status tracking"]
    },
    "6": {
        "name": "Interactive Parking Game",
        "file": "parking_simulator_game.py",
        "description": "Manual click-to-add cars, interactive real-time game",
        "features": ["Click-to-play", "Hover tooltips", "Day comparison"]
    }
}

def print_header():
    """Print project header"""
    print("\n" + "="*80)
    print("🅿️  PARKING LOT DYNAMIC PRICING PROJECT")
    print("="*80)
    print("\n📦 Complete System with RL-Based Pricing Model")
    print("   Project includes: Simulators, Dashboard, Analytics, RL Training\n")

def print_menu():
    """Print selection menu"""
    print("="*80)
    print("SELECT SIMULATOR TO RUN:")
    print("="*80)
    
    for key, sim in SIMULATORS.items():
        print(f"\n{key}. {sim['name']}")
        print(f"   📝 {sim['description']}")
        print(f"   ✓ Features: {', '.join(sim['features'])}")

def run_simulator(choice):
    """Run selected simulator"""
    if choice not in SIMULATORS:
        print("❌ Invalid choice!")
        return
    
    sim = SIMULATORS[choice]
    file_path = sim['file']
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"\n❌ File not found: {file_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"🚀 Launching: {sim['name']}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run([sys.executable, file_path], check=False)
    except Exception as e:
        print(f"❌ Error running simulator: {e}")

def main():
    """Main launcher"""
    print_header()
    
    while True:
        print_menu()
        
        print(f"\n{'='*80}")
        print("0. Exit")
        print(f"{'='*80}\n")
        
        choice = input("Enter your choice (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Thank you for using the Parking Lot Pricing Project!")
            break
        
        run_simulator(choice)
        
        input("\nPress ENTER to return to menu...")
        print("\n" * 2)

if __name__ == "__main__":
    main()
