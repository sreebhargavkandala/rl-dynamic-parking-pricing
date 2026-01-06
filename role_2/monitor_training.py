 
 

import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime


class TrainingMonitor:
    """Monitor training progress in real-time"""
    
    def __init__(self, results_dir="./training_results_optimized"):
        self.results_dir = Path(results_dir)
        self.last_update = 0
        
    def print_banner(self):
        print("\n" + "="*80)
        print(" "*20 + "TRAINING PROGRESS MONITOR")
        print("="*80 + "\n")
    
    def check_status(self):
        """Check current training status"""
        metrics_file = self.results_dir / "training_metrics.json"
        
        if not metrics_file.exists():
            print("⏳ Training not started yet or results directory not found")
            print(f"   Looking in: {self.results_dir}")
            return None
        
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            return metrics
        except:
            return None
    
    def display_metrics(self, metrics):
        """Display training metrics"""
        if metrics is None:
            return
        
        print(" TRAINING METRICS:")
        print(f"  Episodes completed: {metrics.get('total_episodes', '?')}")
        print(f"  Best reward: ${metrics.get('best_reward', 0):.2f}")
        print(f"  Current reward: ${metrics.get('final_reward', 0):.2f}")
        print(f"  Average reward: ${metrics.get('avg_reward', 0):.2f}")
        print(f"  Std deviation: ${metrics.get('std_reward', 0):.2f}")
        
        best_model = metrics.get('best_model_path', 'N/A')
        if best_model != 'N/A':
            best_model = Path(best_model).name
        print(f"  Best model: {best_model}")
        print()
    
    def monitor_loop(self, check_interval=5):
        """Monitor training in a loop"""
        self.print_banner()
        
        print(f" Monitoring training in {self.results_dir.name}/\n")
        print(f"Checking every {check_interval} seconds...")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                metrics = self.check_status()
                
                if metrics:
                    self.display_metrics(metrics)
                    print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print(" Still waiting for training to start...")
                
                time.sleep(check_interval)
                print("-" * 80)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
    
    def load_history(self):
        """Load full training history"""
        history_file = self.results_dir / "training_history.json"
        
        if not history_file.exists():
            print("History file not found")
            return None
        
        with open(history_file) as f:
            return json.load(f)
    
    def analyze_training(self):
        """Analyze training results"""
        history = self.load_history()
        
        if not history:
            return
        
        rewards = history.get('rewards', [])
        
        if not rewards:
            print("No training data yet")
            return
        
        print("\n" + "="*80)
        print(" "*25 + "TRAINING ANALYSIS")
        print("="*80 + "\n")
        
        # Basic stats
        print(" STATISTICS:")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Best: ${max(rewards):.2f}")
        print(f"  Worst: ${min(rewards):.2f}")
        print(f"  Mean: ${np.mean(rewards):.2f}")
        print(f"  Median: ${np.median(rewards):.2f}")
        print(f"  Std: ${np.std(rewards):.2f}")
        
        # Learning progress
        if len(rewards) > 100:
            print("\n LEARNING PROGRESS:")
            first_100 = np.mean(rewards[:100])
            second_100 = np.mean(rewards[100:200]) if len(rewards) > 100 else first_100
            last_100 = np.mean(rewards[-100:])
            
            print(f"  First 100: ${first_100:.2f}")
            print(f"  Last 100: ${last_100:.2f}")
            
            improvement = ((last_100 - first_100) / (first_100 + 1e-6)) * 100
            print(f"  Improvement: {improvement:+.1f}%")
        
        # Convergence
        if len(rewards) > 200:
            print("\n CONVERGENCE:")
            last_50 = rewards[-50:]
            last_50_std = np.std(last_50)
            last_50_mean = np.mean(last_50)
            
            print(f"  Last 50 mean: ${last_50_mean:.2f}")
            print(f"  Last 50 std: ${last_50_std:.2f}")
            print(f"  Stability: {100 - (last_50_std/last_50_mean*100):.1f}%")
        
        print("\n" + "="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--monitor", action="store_true", help="Monitor training live")
    parser.add_argument("--analyze", action="store_true", help="Analyze completed training")
    parser.add_argument("--dir", default="./training_results_optimized", help="Results directory")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.dir)
    
    if args.monitor:
        monitor.monitor_loop(args.interval)
    elif args.analyze:
        monitor.analyze_training()
    else:
        # Default: show current status
        metrics = monitor.check_status()
        monitor.display_metrics(metrics)


if __name__ == "__main__":
    main()
