"""
HOW TO RUN THE PARKING LOT PRICING PROJECT
===========================================
Complete guide to running all simulators and dashboards
"""

INSTALLATION_REQUIREMENTS = """
INSTALLATION & SETUP
====================

1. VERIFY PYTHON IS INSTALLED:
   Open Command Prompt/PowerShell and type:
   python --version
   
   Should show Python 3.7 or higher

2. INSTALL REQUIRED LIBRARIES (one time only):
   
   Open PowerShell in the project folder and run:
   
   pip install pygame numpy matplotlib
   
   Or install individually:
   - pip install pygame
   - pip install numpy
   - pip install matplotlib

3. VERIFY INSTALLATION:
   python -m pip list
   (Look for pygame, numpy, matplotlib in the list)
"""

QUICK_START = """
QUICK START - 3 SECONDS TO RUN
==============================

OPTION A: USE THE LAUNCHER (EASIEST) ⭐ RECOMMENDED
-----------------------------------------------------
Open PowerShell in project folder and type:

python PROJECT_LAUNCHER.py

Then select from menu:
  1 = Main RL Simulator (Best for faculty demo)
  2 = Day-by-Day Simulator
  3 = Realistic Parking Demo
  4 = Demo Simulator
  5 = Real-Time Dashboard
  6 = Interactive Game
  0 = Exit


OPTION B: RUN DIRECTLY (CHOOSE ONE)
------------------------------------

For Faculty Presentation (BEST):
  python rl_integrated_simulator.py

For Step-by-Step Analysis:
  python day_by_day_simulator.py

For Beautiful Visualization:
  python realistic_parking_demo.py

For Algorithm Breakdown:
  python demo_simulator.py

For Real-Time Metrics:
  python dashboard.py

For Interactive Play:
  python parking_simulator_game.py
"""

DETAILED_INSTRUCTIONS = """
DETAILED RUN INSTRUCTIONS
==========================

STEP 1: OPEN COMMAND PROMPT / POWERSHELL
-----------------------------------------
Windows:
- Press Windows Key + R
- Type: powershell
- Press Enter

Or right-click in the project folder and select:
"Open PowerShell window here"

STEP 2: NAVIGATE TO PROJECT FOLDER
-----------------------------------
If not already there, type:
cd "c:\Users\harsh\Downloads\RL_Project\rl-dynamic-parking-pricing"

STEP 3: RUN THE LAUNCHER
------------------------
Type:
python PROJECT_LAUNCHER.py

Then press Enter

STEP 4: SELECT SIMULATOR
------------------------
When menu appears, type a number (1-6) and press Enter

STEP 5: ENJOY!
--------------
The simulator will launch in a window
- Close window or press ESC to exit
- Return to menu and try another option


WHAT TO EXPECT
==============

When running a simulator:

✅ A window will open with the parking lot
✅ Cars will appear and park in real-time
✅ Prices will update based on occupancy
✅ Revenue tracking will show at top/bottom
✅ Click NEXT DAY button to progress (some simulators)
✅ Hover over cars to see details
✅ Charts and graphs will display metrics

Close the window to return to menu
"""

SIMULATOR_DETAILS = """
SIMULATOR DETAILS & CONTROLS
============================

1️⃣ RL-INTEGRATED SIMULATOR (RECOMMENDED)
==========================================
Command: python rl_integrated_simulator.py
Duration: 5-10 minutes recommended

Controls:
- SPACE = Pause/Resume
- ↑↓ (Arrows) = Adjust speed (some versions)
- ESC = Exit

What to Watch:
✓ Click NEXT DAY button when day completes
✓ Watch RL model improve pricing each day
✓ Occupancy stays at 60% minimum
✓ Revenue increases as model learns
✓ Price changes based on time/weather/occupancy

Key Panel (Right Side):
- Shows current price calculation
- Shows learning rate and exploration
- Shows days trained


2️⃣ DAY-BY-DAY SIMULATOR
=========================
Command: python day_by_day_simulator.py
Duration: 2-3 minutes per day

Controls:
- SPACE = Pause/Resume
- Click ► NEXT DAY button = Progress to next day
- ESC = Exit

What to Watch:
✓ One complete day per session (6 AM - 6 PM)
✓ See how prices change throughout day
✓ Compare with previous day (shown in corner)
✓ Slow motion simulation - easy to follow

Key Features:
- Previous day revenue shown
- Car count tracked
- Average price comparison
- Time-based demand patterns


3️⃣ REALISTIC PARKING DEMO
============================
Command: python realistic_parking_demo.py
Duration: 5-10 minutes

Controls:
- SPACE = Pause/Resume
- ↑↓ (Arrows) = Speed control
- ESC = Exit

What to Watch:
✓ Beautiful 3D-style car graphics
✓ Real parking lot layout
✓ Slow motion (0.3x speed) - very clear
✓ Entrance and exit gates
✓ Cars with windows, wheels, license plates

Best For:
- Understanding how system works visually
- Showing to non-technical audience
- Understanding parking dynamics


4️⃣ DEMO SIMULATOR (ALGORITHM EXPLANATION)
=============================================
Command: python demo_simulator.py
Duration: 3-5 minutes

Controls:
- SPACE = Pause/Resume
- Hover mouse over cars = See details
- ESC = Exit

What to Watch:
✓ Algorithm breakdown on right side
✓ Shows each pricing factor:
  - Base Price
  - Occupancy Factor
  - Peak Hour Multiplier
  - Weather Effect
  - Weekend Multiplier
✓ Final calculated price shown
✓ Perfect for explaining algorithm

Best For:
- Faculty technical presentation
- Explaining pricing algorithm
- Understanding each pricing factor


5️⃣ REAL-TIME DASHBOARD
==========================
Command: python dashboard.py
Duration: Continuous monitoring

Controls:
- ESC = Exit
- No pause/resume (continuous data collection)

What to Watch:
✓ Real-time price trends (chart)
✓ Daily revenue bars (chart)
✓ Occupancy percentage (chart)
✓ Key metrics cards:
  - Current pricing
  - Occupancy stats
  - Revenue tracking
  - Volume metrics
✓ RL model status:
  - Training days
  - Learning rate
  - Exploration rate

Best For:
- Project monitoring
- Analytics review
- Performance tracking
- Long-term observation


6️⃣ INTERACTIVE GAME
======================
Command: python parking_simulator_game.py
Duration: 5-10 minutes

Controls:
- Click on "ENTER CARS" button = Add cars manually
- Hover over parked cars = See details
- Watch auto-generated events
- SPACE = Pause/Resume
- ESC = Exit

What to Watch:
✓ Click to add cars to parking lot
✓ See prices assigned to each car
✓ Track daily revenue in real-time
✓ Compare with previous day metrics
✓ Interactive parking management

Best For:
- Casual exploration
- Interactive learning
- Understanding system dynamics
- Fun demonstration
"""

TROUBLESHOOTING = """
TROUBLESHOOTING
===============

PROBLEM: "python: command not found"
SOLUTION:
- Python may not be installed
- Or Python not in PATH
- Install Python from python.org
- During installation, CHECK: "Add Python to PATH"

PROBLEM: "ModuleNotFoundError: No module named 'pygame'"
SOLUTION:
- pygame not installed
- Run: pip install pygame
- Then retry: python FILENAME.py

PROBLEM: "Window doesn't appear"
SOLUTION:
- Wait 2-3 seconds, it may be loading
- Check taskbar for open windows
- Try pressing ESC and rerunning

PROBLEM: "Game is frozen/stuck"
SOLUTION:
- Press SPACE to check if paused
- Press ESC to exit
- Restart the simulator

PROBLEM: "Charts not showing in dashboard"
SOLUTION:
- This is normal initially, wait for data
- Charts update every 2 seconds once data available
- Continue running, charts will appear

PROBLEM: "Slow/laggy performance"
SOLUTION:
- Close other applications
- Try different simulator (lighter version)
- Reduce screen resolution if needed
- Most simulators run smooth on modern PCs

PROBLEM: "matplotlib issues"
SOLUTION:
- Run: pip install --upgrade matplotlib
- If still issues, run dashboard without charts
"""

RECOMMENDED_SEQUENCE = """
RECOMMENDED VIEWING SEQUENCE
============================

FOR FACULTY PRESENTATION (15 minutes):
1. Show Project Documentation (5 min)
   python PROJECT_DOCUMENTATION.py

2. Run RL Simulator (10 min)
   python rl_integrated_simulator.py
   - Start a fresh day
   - Click NEXT DAY 5-10 times
   - Watch prices improve
   - Point out learning rate decreasing
   - Show revenue increasing

3. Optional: Show Dashboard
   python dashboard.py
   - Live metrics
   - Learning progress


FOR STUDENT LEARNING (30 minutes):
1. Start with Dashboard (5 min)
   python dashboard.py
   - Understand what's being tracked

2. Run Day-by-Day Simulator (10 min)
   python day_by_day_simulator.py
   - Step through multiple days
   - Understand daily patterns

3. Run Demo Simulator (10 min)
   python demo_simulator.py
   - See algorithm breakdown
   - Understand pricing factors

4. Explore Realistic Demo (5 min)
   python realistic_parking_demo.py
   - Appreciate visual aspect


FOR QUICK DEMO (5 minutes):
   python rl_integrated_simulator.py
   - Click NEXT DAY 3 times
   - Watch prices and revenue change
   - Explain Q-Learning happening behind scenes


FOR DETAILED ANALYSIS (1+ hour):
1. Run each simulator once
2. Use dashboard to monitor
3. Compare metrics across runs
4. Modify simulator code if desired
"""

KEYBOARD_SHORTCUTS = """
UNIVERSAL KEYBOARD SHORTCUTS
=============================

ESC = Exit any simulator (return to menu if using launcher)
SPACE = Pause/Resume (most simulators)

RL-INTEGRATED SIMULATOR:
- SPACE = Pause/Resume the simulation

DAY-BY-DAY SIMULATOR:
- SPACE = Pause/Resume
- Click ► NEXT DAY = Progress day

REALISTIC PARKING DEMO:
- SPACE = Pause/Resume
- ↑ Arrow = Speed up
- ↓ Arrow = Speed down

MOUSE ACTIONS (All simulators):
- Hover over cars = See car details
- Click buttons = Trigger actions
"""

FILE_SUMMARY = """
PROJECT FILES SUMMARY
=====================

MAIN SIMULATORS:
✅ rl_integrated_simulator.py - BEST FOR DEMO
✅ day_by_day_simulator.py - Step-by-step
✅ realistic_parking_demo.py - Beautiful graphics
✅ demo_simulator.py - Algorithm explanation
✅ parking_simulator_game.py - Interactive play

MONITORING:
✅ dashboard.py - Real-time metrics & charts

MANAGEMENT:
✅ PROJECT_LAUNCHER.py - Main menu launcher
✅ PROJECT_DOCUMENTATION.py - This documentation

SUPPORTING FILES:
- role_2/ - RL algorithms (PPO, SAC, A2C, DDPG)
- role_1/ - Environment and metrics
- role_4/ - Evaluation and baselines
- training_results/ - Saved training data
"""

def print_complete_guide():
    print("\n" + "="*80)
    print("HOW TO RUN THE PARKING LOT PRICING PROJECT")
    print("="*80)
    
    print(INSTALLATION_REQUIREMENTS)
    print("\n" + "="*80)
    print(QUICK_START)
    print("\n" + "="*80)
    print(DETAILED_INSTRUCTIONS)
    print("\n" + "="*80)
    print(SIMULATOR_DETAILS)
    print("\n" + "="*80)
    print(TROUBLESHOOTING)
    print("\n" + "="*80)
    print(RECOMMENDED_SEQUENCE)
    print("\n" + "="*80)
    print(KEYBOARD_SHORTCUTS)
    print("\n" + "="*80)
    print(FILE_SUMMARY)
    print("\n" + "="*80)
    print("READY TO RUN!")
    print("="*80)
    print("\n🎬 Next step: python PROJECT_LAUNCHER.py\n")

if __name__ == "__main__":
    print_complete_guide()
