"""
Parking Simulator Game Launcher
================================
Easy launcher to start the interactive parking lot simulator game.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_simulator():
    """Run the parking simulator game"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║              🚗 DYNAMIC PARKING LOT SIMULATOR GAME 🅿️                     ║
    ║                                                                            ║
    ║  An interactive real-time simulator for dynamic parking pricing!          ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    FEATURES:
    ═════════════════════════════════════════════════════════════════════════════
    
    ✅ Interactive Parking Lot Management
       └─ Visual 5x8 parking grid with real-time updates
       └─ Watch cars enter and leave dynamically
       └─ Click "ADD CAR" button to spawn new vehicles
    
    ✅ Dynamic Pricing System
       └─ Prices adjust based on current occupancy
       └─ Peak hour multipliers (9-12, 12-14, 17-20)
       └─ Price range: $2-20 per hour
    
    ✅ Car Details & Tooltips
       └─ Hover over any parked car to see details
       └─ Display: Car ID, assigned price, duration, time parked
       └─ Click car for quick info popup
    
    ✅ Revenue Tracking
       └─ Real-time daily revenue counter
       └─ Automatic previous day comparison
       └─ Revenue history saved to file
    
    ✅ Game Controls
       └─ Speed up (↑) / Slow down (↓)
       └─ Pause/Resume (SPACE)
       └─ Save revenue (S)
       └─ Start new day (R)
    
    GAME MECHANICS:
    ═════════════════════════════════════════════════════════════════════════════
    
    1. VEHICLE ENTRY
       • Click the "ADD CAR" button to spawn a new vehicle
       • Each car gets a random parking duration (30 min - 3 hours)
       • System assigns dynamic price based on occupancy
    
    2. DYNAMIC PRICING
       • Formula: Price = Base + (Occupancy² × Range) × Peak Multiplier
       • Base price: $5.00
       • Occupancy factor: Non-linear (quadratic)
       • Peak hour multiplier: 1.3x during busy hours
    
    3. REVENUE GENERATION
       • Each parked car generates revenue at their assigned price
       • Total displayed in real-time
       • Compare with previous day's earnings
    
    4. OCCUPANCY TRACKING
       • Monitor parking lot utilization
       • See available spots
       • Track average occupancy
    
    TIPS FOR BEST RESULTS:
    ═════════════════════════════════════════════════════════════════════════════
    
    📌 Try to maintain 60-80% occupancy for optimal revenue
    📌 Watch for peak hours (9-12, 12-14, 17-20) for higher demand
    📌 Speed up the simulation to see trends faster (press ↑)
    📌 Run multiple days to build revenue history
    📌 Check if your pricing strategy beats yesterday's revenue!
    
    KEYBOARD SHORTCUTS:
    ═════════════════════════════════════════════════════════════════════════════
    
    CLICK                  Add car at gate / View car details
    SPACE                  Pause / Resume simulation
    ↑ / ↓                  Increase / Decrease simulation speed
    S                      Save current day's revenue
    R                      Start new day (saves previous)
    ESC                    Close game
    
    ═════════════════════════════════════════════════════════════════════════════
    
    Starting simulator...
    """)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    simulator_file = script_dir / "parking_simulator_game.py"
    
    if not simulator_file.exists():
        print(f"❌ Error: Could not find parking_simulator_game.py")
        print(f"   Expected location: {simulator_file}")
        return 1
    
    try:
        # Run the simulator
        result = subprocess.run(
            [sys.executable, str(simulator_file)],
            cwd=str(script_dir)
        )
        return result.returncode
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulator interrupted by user")
        return 0
    
    except Exception as e:
        print(f"❌ Error running simulator: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_simulator()
    sys.exit(exit_code)
