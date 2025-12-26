# 🚗 RL Dynamic Parking Pricing - Organized Project

## 📁 Project Structure

```
rl-dynamic-parking-pricing/
│
├── config/                          # Project Configuration & Setup
│   ├── PROJECT_LAUNCHER.py          # 🎯 Main menu - Start here!
│   ├── PROJECT_DOCUMENTATION.py     # 📖 Full documentation
│   ├── HOW_TO_RUN.py               # Quick start guide
│   └── RUN_GUIDE.py                # Detailed instructions
│
├── simulators/                      # 🎮 All Parking Lot Simulators
│   ├── rl_integrated_simulator.py  # ⭐ Main: Q-Learning pricing (RECOMMENDED)
│   ├── day_by_day_simulator.py     # Daily progression with comparison
│   ├── realistic_parking_demo.py   # 3D-style visualization with slow motion
│   ├── demo_simulator.py           # Algorithm breakdown visualization
│   ├── parking_simulator_game.py   # Interactive click-to-play game
│   ├── automated_game.py           # Automated visual simulator
│   └── automated_simulator.py      # Terminal-based multi-week simulation
│
├── dashboard/                       # 📊 Real-Time Monitoring
│   └── dashboard.py                # Live metrics & analytics dashboard
│
├── monitoring/                      # 📈 Analysis & Evaluation
│   ├── analyze_results.py          # Performance analysis
│   └── show_training_results.py    # Training metrics viewer
│
├── utils/                          # 🔧 Utility Scripts
│   ├── run_simulator.py            # Run single simulator
│   └── run_all.py                  # Run all simulators
│
├── data/                           # 💾 Results & Data Files
│   ├── revenue_history.json        # Simulation revenue data
│   ├── simulation_results_visual.json
│   └── revenue_history_automated.json
│
├── role_1/                         # 🔬 Environment & Metrics
│   ├── env.py                      # Parking lot environment
│   ├── reward_function.py          # Reward calculation
│   ├── state_action_documentation.py
│   └── metrics.py
│
├── role_2/                         # 🤖 RL Models & Algorithms
│   ├── ppo.py                      # Proximal Policy Optimization
│   ├── sac.py                      # Soft Actor-Critic
│   ├── a2c.py                      # Actor-Critic
│   ├── ddpg.py                     # Deep Deterministic Policy Gradient
│   └── networks.py                 # Neural network architectures
│
└── role_3/ & role_4/              # Additional Research Modules
```

## 🚀 Quick Start

### Option 1: Use the Project Launcher (Easiest)
```bash
cd config
python PROJECT_LAUNCHER.py
```
This opens an interactive menu with all available simulators and tools.

### Option 2: Run the Main Simulator (Recommended)
```bash
cd simulators
python rl_integrated_simulator.py
```
Features:
- Q-Learning RL pricing model
- Maintains 60% minimum occupancy
- Dynamic pricing based on demand
- Day-by-day learning progression
- Real-time visualization

### Option 3: Run the Dashboard (Monitoring)
```bash
cd dashboard
python dashboard.py
```
Real-time tracking of:
- Pricing metrics
- Revenue trends
- Occupancy levels
- RL model performance

## 📚 Available Simulators

| Simulator | Purpose | Start Time |
|-----------|---------|-----------|
| **rl_integrated_simulator.py** | Main RL-based simulator with Q-Learning | ~2 minutes |
| **day_by_day_simulator.py** | Step-by-step daily progression | Interactive |
| **realistic_parking_demo.py** | Beautiful 3D-style visualization | ~3 minutes |
| **demo_simulator.py** | Algorithm breakdown with annotations | ~5 minutes |
| **parking_simulator_game.py** | Interactive click-to-add cars | Interactive |
| **automated_game.py** | Automated with visual feedback | ~2 minutes |
| **automated_simulator.py** | Terminal-based multi-week run | ~1 minute |

## 🎯 Features

✅ **Reinforcement Learning (Q-Learning)**
- Learns optimal pricing over time
- Epsilon-greedy exploration/exploitation
- Daily training and improvement

✅ **Realistic Simulation**
- Dynamic pricing adjustments
- Multiple pricing factors (time, weather, occupancy)
- Demand variation throughout day

✅ **Real-time Monitoring**
- Live dashboard with graphs
- Revenue and occupancy tracking
- Price trend analysis

✅ **Multiple Visualization Modes**
- 3D-style parking lot view
- Algorithm breakdown display
- Interactive game interface
- Terminal output

✅ **Occupancy Management**
- Maintains 60% minimum occupancy target
- Automatic price adjustments
- Demand forecasting

## 📊 Key Metrics

- **Revenue Optimization**: Learns to maximize parking revenue
- **Occupancy Target**: Maintains 60% minimum occupancy
- **Price Stability**: Smooth pricing transitions
- **Training Convergence**: 5-10 days to optimal policy
- **Peak Hour Management**: Dynamic pricing during busy hours

## 🔬 Technical Details

**Machine Learning:**
- Algorithm: Q-Learning with discrete state/action spaces
- States: (occupancy_level, hour_period, weather)
- Actions: 5 price levels ($5-$25)
- Learning Rate: 0.1 (decays over time)
- Discount Factor: 0.95

**Simulation Parameters:**
- Parking Spaces: 50
- Simulation Duration: 7+ days
- Update Frequency: Real-time
- Random Seed: 42 (reproducible)

## 📖 Documentation

For detailed information, run:
```bash
cd config
python PROJECT_DOCUMENTATION.py
```

## 🎓 Use Cases

- **Faculty Presentations**: Run `rl_integrated_simulator.py`
- **Learning Demo**: Run `realistic_parking_demo.py`
- **Interactive Demo**: Run `parking_simulator_game.py`
- **Data Analysis**: Run `monitoring/analyze_results.py`
- **Monitoring System**: Run `dashboard/dashboard.py`

## ⚙️ System Requirements

- Python 3.7+
- pygame 2.0+
- numpy
- matplotlib
- json (built-in)

## 📝 File Organization

Files are now organized into functional folders:
- **config/** - Everything to get started
- **simulators/** - All simulation variants
- **dashboard/** - Monitoring systems
- **monitoring/** - Analysis tools
- **utils/** - Helper scripts
- **data/** - Results and outputs
- **role_1, 2, 3, 4/** - RL implementation modules

## 🐛 Troubleshooting

**Dashboard not updating?**
- Make sure it's running: `python dashboard.py`
- Check terminal for errors

**Simulator running too fast/slow?**
- Speed is configurable in each simulator file
- Look for `SPEED_FACTOR` or `FPS` variables

**Prices not changing?**
- Model needs training time (5-10 days)
- Check RL model in rl_integrated_simulator.py

## 📞 Support

For complete documentation: `config/PROJECT_DOCUMENTATION.py`
For quick start guide: `config/HOW_TO_RUN.py`

---

**Status**: ✅ Production Ready | **Last Updated**: December 2025
