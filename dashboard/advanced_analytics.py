"""
ADVANCED ANALYTICS DASHBOARD
Real-time RL pricing analytics with advanced visualizations
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict, deque

class AdvancedAnalytics:
    """Advanced metrics calculation for parking pricing"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.rolling_window = 50
        
    def calculate_price_elasticity(self, prices, occupancies):
        """Calculate price elasticity of demand"""
        if len(prices) < 2 or len(occupancies) < 2:
            return 0.0
        
        prices = np.array(prices)
        occupancies = np.array(occupancies)
        
        # Calculate percentage changes
        price_changes = np.diff(prices) / prices[:-1]
        occupancy_changes = np.diff(occupancies) / occupancies[:-1]
        
        # Elasticity = % change in quantity / % change in price
        elasticity = np.mean(occupancy_changes) / np.mean(np.abs(price_changes) + 1e-6)
        return elasticity
    
    def detect_anomalies(self, prices):
        """Detect sudden price changes (anomalies)"""
        if len(prices) < 3:
            return []
        
        prices = np.array(prices)
        mean = np.mean(prices)
        std = np.std(prices)
        
        anomalies = []
        for i, price in enumerate(prices):
            if abs(price - mean) > self.anomaly_threshold * std:
                anomalies.append({
                    'index': i,
                    'price': float(price),
                    'deviation': float((price - mean) / std)
                })
        
        return anomalies
    
    def calculate_peak_hours(self, occupancies, window=12):
        """Identify peak occupancy hours"""
        if len(occupancies) < window:
            return []
        
        occupancies = np.array(occupancies)
        rolling_avg = np.convolve(occupancies, np.ones(window)/window, mode='valid')
        
        peaks = []
        for i in range(len(rolling_avg)):
            if rolling_avg[i] > 0.85:
                peaks.append({
                    'hour': i,
                    'occupancy': float(rolling_avg[i])
                })
        
        return peaks
    
    def calculate_revenue_efficiency(self, prices, occupancies):
        """Revenue per occupancy unit"""
        if len(prices) == 0 or len(occupancies) == 0:
            return 0.0
        
        prices = np.array(prices)
        occupancies = np.array(occupancies)
        
        revenue = prices * occupancies
        efficiency = np.mean(revenue) / (np.mean(prices) + 1e-6)
        
        return float(efficiency)
    
    def calculate_stability_score(self, prices):
        """Score pricing stability (0-100)"""
        if len(prices) < 2:
            return 100.0
        
        prices = np.array(prices)
        volatility = np.std(prices) / (np.mean(prices) + 1e-6)
        
        # Convert volatility to stability score (lower volatility = higher score)
        stability = max(0, 100 - (volatility * 100))
        return float(stability)
    
    def generate_report(self, prices, occupancies, rewards):
        """Generate comprehensive analytics report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'price_elasticity': self.calculate_price_elasticity(prices, occupancies),
            'anomalies': self.detect_anomalies(prices),
            'peak_hours': self.calculate_peak_hours(occupancies),
            'revenue_efficiency': self.calculate_revenue_efficiency(prices, occupancies),
            'stability_score': self.calculate_stability_score(prices),
            'statistics': {
                'avg_price': float(np.mean(prices)) if prices else 0,
                'min_price': float(np.min(prices)) if prices else 0,
                'max_price': float(np.max(prices)) if prices else 0,
                'price_std': float(np.std(prices)) if prices else 0,
                'avg_occupancy': float(np.mean(occupancies)) if occupancies else 0,
                'occupancy_std': float(np.std(occupancies)) if occupancies else 0,
                'avg_reward': float(np.mean(rewards)) if rewards else 0,
                'total_episodes_analyzed': len(rewards),
            }
        }
        
        return report
    
    def save_report(self, report, filepath):
        """Save analytics report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to {filepath}")
    
    def export_training_data(self, prices, occupancies, rewards, filepath):
        """Export training data for further analysis"""
        data = {
            'prices': [float(p) for p in prices],
            'occupancies': [float(o) for o in occupancies],
            'rewards': [float(r) for r in rewards],
            'timestamps': [i for i in range(len(prices))]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Data exported to {filepath}")


class PerformanceTracker:
    """Track and analyze RL agent performance"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_losses = deque(maxlen=window_size)
        
    def update(self, reward, length, loss=None):
        """Update with episode data"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.episode_losses.append(loss)
    
    def get_improvement_rate(self):
        """Calculate learning improvement rate"""
        if len(self.episode_rewards) < 2:
            return 0.0
        
        recent = list(self.episode_rewards)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        improvement = (second_half - first_half) / (first_half + 1e-6)
        return float(improvement)
    
    def get_statistics(self):
        """Get performance statistics"""
        if not self.episode_rewards:
            return {}
        
        rewards = list(self.episode_rewards)
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'improvement_rate': self.get_improvement_rate(),
            'total_episodes': len(rewards),
        }
    
    def is_converged(self, threshold=0.05):
        """Check if agent has converged"""
        if len(self.episode_rewards) < 20:
            return False
        
        recent = list(self.episode_rewards)[-20:]
        variance = np.std(recent) / (np.mean(recent) + 1e-6)
        
        return variance < threshold


class DemandAnalyzer:
    """Analyze parking demand patterns"""
    
    def __init__(self):
        self.hourly_occupancy = defaultdict(list)
        self.hourly_price = defaultdict(list)
        
    def add_observation(self, hour, occupancy, price):
        """Add hourly observation"""
        self.hourly_occupancy[hour].append(occupancy)
        self.hourly_price[hour].append(price)
    
    def get_peak_hours(self):
        """Get peak demand hours"""
        peaks = []
        for hour, occupancies in self.hourly_occupancy.items():
            avg_occ = np.mean(occupancies)
            if avg_occ > 0.75:
                peaks.append({
                    'hour': hour,
                    'avg_occupancy': float(avg_occ),
                    'avg_price': float(np.mean(self.hourly_price[hour]))
                })
        
        return sorted(peaks, key=lambda x: x['avg_occupancy'], reverse=True)
    
    def get_demand_distribution(self):
        """Get hourly demand distribution"""
        distribution = {}
        for hour in sorted(self.hourly_occupancy.keys()):
            occupancies = self.hourly_occupancy[hour]
            distribution[hour] = {
                'mean': float(np.mean(occupancies)),
                'std': float(np.std(occupancies)),
                'min': float(np.min(occupancies)),
                'max': float(np.max(occupancies)),
                'count': len(occupancies)
            }
        
        return distribution


def create_analysis_summary(prices, occupancies, rewards):
    """Create a summary analysis"""
    analytics = AdvancedAnalytics()
    report = analytics.generate_report(prices, occupancies, rewards)
    
    print("\n" + "="*80)
    print("ADVANCED ANALYTICS REPORT")
    print("="*80)
    print(f"\n📊 STATISTICS:")
    print(f"  Average Price: ${report['statistics']['avg_price']:.2f}")
    print(f"  Price Range: ${report['statistics']['min_price']:.2f} - ${report['statistics']['max_price']:.2f}")
    print(f"  Price Volatility: {report['statistics']['price_std']:.2f}")
    print(f"  Average Occupancy: {report['statistics']['avg_occupancy']*100:.1f}%")
    print(f"  Occupancy Std: {report['statistics']['occupancy_std']*100:.1f}%")
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"  Price Elasticity: {report['price_elasticity']:.3f}")
    print(f"  Revenue Efficiency: {report['revenue_efficiency']:.3f}")
    print(f"  Stability Score: {report['stability_score']:.1f}/100")
    
    print(f"\n⚠️ ANOMALIES DETECTED: {len(report['anomalies'])}")
    for anomaly in report['anomalies'][:5]:
        print(f"  - At step {anomaly['index']}: ${anomaly['price']:.2f} "
              f"({anomaly['deviation']:.2f}σ deviation)")
    
    print(f"\n📈 PEAK HOURS: {len(report['peak_hours'])}")
    for peak in report['peak_hours'][:5]:
        print(f"  - Hour {peak['hour']}: {peak['occupancy']*100:.1f}%")
    
    print("\n" + "="*80 + "\n")
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Advanced Analytics Module - Parking Pricing System")
    print("Use this module for detailed performance analysis\n")
    
    # Example data
    prices = [5.0, 8.5, 12.0, 11.5, 8.75, 6.0, 9.0, 13.5, 15.0, 12.0]
    occupancies = [0.55, 0.65, 0.78, 0.85, 0.78, 0.62, 0.70, 0.88, 0.92, 0.81]
    rewards = [125.5, 225.8, 345.2, 312.5, 245.6, 185.2, 245.0, 380.5, 420.2, 325.5]
    
    report = create_analysis_summary(prices, occupancies, rewards)
