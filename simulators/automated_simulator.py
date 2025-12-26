"""
Automated Parking Lot Simulator - Multi-Week Run
================================================

Automatically run the parking lot simulator for multiple weeks with:
- Weather effects
- Weekend variations
- Holiday impacts
- Dynamic pricing adjustments
- Real-time daily revenue tracking
- Weekly and overall summaries
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time


@dataclass
class DailyStats:
    """Daily statistics"""
    date: str
    day_name: str
    revenue: float
    cars: int
    avg_price: float
    occupancy: float
    weather: str
    is_weekend: bool
    is_holiday: bool
    peak_hours: str


class WeatherSimulator:
    """Simulates weather effects on parking demand"""
    
    def __init__(self):
        self.weather_types = ["Sunny", "Rainy", "Cloudy", "Snowy", "Foggy"]
        self.weather_effects = {
            "Sunny": 1.0,      # Normal demand
            "Rainy": 0.85,     # Reduced demand
            "Cloudy": 0.95,    # Slight reduction
            "Snowy": 0.7,      # Much less demand
            "Foggy": 0.9       # Slight reduction
        }
    
    def get_weather_and_effect(self, date: datetime) -> Tuple[str, float]:
        """Get random weather and demand effect"""
        weather = random.choices(
            self.weather_types,
            weights=[0.4, 0.2, 0.2, 0.1, 0.1]
        )[0]
        effect = self.weather_effects[weather]
        return weather, effect


class HolidaySimulator:
    """Simulates holiday effects"""
    
    def __init__(self):
        # Common holidays (month-day format)
        self.holidays = {
            "01-01": ("New Year", 0.6),
            "12-25": ("Christmas", 0.5),
            "07-04": ("Independence Day", 0.7),
            "11-27": ("Thanksgiving", 0.8),
            "05-27": ("Memorial Day", 0.8),
            "09-01": ("Labor Day", 0.8),
        }
    
    def get_holiday_info(self, date: datetime) -> Tuple[bool, str, float]:
        """Check if day is holiday and get effect"""
        date_key = date.strftime("%m-%d")
        
        if date_key in self.holidays:
            name, effect = self.holidays[date_key]
            return True, name, effect
        return False, "", 1.0


class AdvancedPricingAgent:
    """Advanced pricing with weather, time, occupancy, etc."""
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 25.0
        self.min_price = 1.5
    
    def calculate_price(self, occupancy: float, hour: int, 
                       weather_effect: float, is_weekend: bool, 
                       holiday_effect: float) -> float:
        """
        Calculate price with multiple factors:
        - Occupancy (non-linear)
        - Time of day (peak hours)
        - Weather (demand effect)
        - Weekend (higher demand)
        - Holiday (special effect)
        """
        
        # Occupancy factor (non-linear)
        occupancy_factor = occupancy ** 2
        
        # Peak hours: 8-10, 12-14, 17-20 (weekdays), 10-14, 18-21 (weekends)
        if is_weekend:
            peak_hours = [10, 11, 12, 13, 14, 18, 19, 20, 21]
        else:
            peak_hours = [8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
        
        peak_multiplier = 1.4 if hour in peak_hours else 1.0
        
        # Weekend premium
        weekend_multiplier = 1.15 if is_weekend else 1.0
        
        # Calculate base price from occupancy
        price = self.base_price + (occupancy_factor * (self.max_price - self.base_price))
        
        # Apply all multipliers
        price = price * peak_multiplier * weekend_multiplier * weather_effect * holiday_effect
        
        # Clamp to min/max
        price = max(self.min_price, min(self.max_price, price))
        
        return round(price, 2)


class AutomatedParkingSimulator:
    """Automated multi-week parking simulator"""
    
    def __init__(self, weeks: int = 1):
        self.weeks = weeks
        self.days = weeks * 7
        self.weather_sim = WeatherSimulator()
        self.holiday_sim = HolidaySimulator()
        self.pricing_agent = AdvancedPricingAgent()
        
        self.daily_stats: List[DailyStats] = []
        self.start_date = datetime.now()
        self.revenue_history: Dict[str, Dict] = self._load_history()
        
        # Simulation parameters
        self.base_daily_cars = random.randint(40, 80)  # Random base
        self.lot_capacity = 40
    
    def _load_history(self) -> Dict:
        """Load existing revenue history"""
        history_file = Path("revenue_history_automated.json")
        if history_file.exists():
            try:
                with open(history_file) as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_history(self) -> None:
        """Save revenue history"""
        with open("revenue_history_automated.json", 'w') as f:
            json.dump(self.revenue_history, f, indent=2)
    
    def simulate_day(self, date: datetime) -> DailyStats:
        """Simulate a single day"""
        
        # Get date info
        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")
        is_weekend = date.weekday() >= 5
        
        # Get weather
        weather, weather_effect = self.weather_sim.get_weather_and_effect(date)
        
        # Get holiday info
        is_holiday, holiday_name, holiday_effect = self.holiday_sim.get_holiday_info(date)
        
        # Simulate cars and pricing throughout the day
        total_revenue = 0.0
        total_cars = 0
        prices = []
        occupancies = []
        
        # Simulate 24 hours
        for hour in range(24):
            # Calculate current occupancy (varies by hour)
            if is_weekend:
                # Weekend pattern: peaks at 11 and 18
                base_occupancy = (
                    0.3 * np.sin((hour - 11) * np.pi / 24) + 
                    0.2 * np.sin((hour - 18) * np.pi / 24) + 0.5
                )
            else:
                # Weekday pattern: peaks at 9 and 17
                base_occupancy = (
                    0.35 * np.sin((hour - 9) * np.pi / 24) + 
                    0.25 * np.sin((hour - 17) * np.pi / 24) + 0.4
                )
            
            # Apply weather effect to occupancy
            occupancy = max(0.0, min(1.0, base_occupancy * weather_effect * holiday_effect))
            occupancies.append(occupancy)
            
            # Calculate price
            price = self.pricing_agent.calculate_price(
                occupancy, hour, weather_effect, is_weekend, holiday_effect
            )
            prices.append(price)
            
            # Simulate cars entering at this hour
            # More cars when weather is good and prices are reasonable
            cars_this_hour = max(0, int(
                (5 + occupancy * 8) * weather_effect * (1 if hour in [9, 12, 18] else 0.7)
            ))
            
            if random.random() > (1 - weather_effect):
                cars_this_hour += 1
            
            total_cars += cars_this_hour
            total_revenue += cars_this_hour * price
        
        # Calculate statistics
        avg_price = np.mean(prices) if prices else 0
        avg_occupancy = np.mean(occupancies) if occupancies else 0
        
        # Peak hours info
        if is_weekend:
            peak_hours = "10-14, 18-21"
        else:
            peak_hours = "8-10, 12-14, 17-20"
        
        # Create daily stats
        stats = DailyStats(
            date=date_str,
            day_name=day_name,
            revenue=round(total_revenue, 2),
            cars=total_cars,
            avg_price=round(avg_price, 2),
            occupancy=round(avg_occupancy * 100, 1),
            weather=weather,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            peak_hours=peak_hours
        )
        
        return stats
    
    def run_simulation(self) -> None:
        """Run the full multi-week simulation"""
        
        print("\n" + "="*80)
        print("🚗 AUTOMATED PARKING LOT SIMULATOR - MULTI-WEEK RUN")
        print("="*80)
        print(f"\n📅 Running simulation for {self.weeks} weeks ({self.days} days)")
        print(f"⏰ Start Date: {self.start_date.strftime('%A, %B %d, %Y')}\n")
        
        # Run each day
        print("Running simulation...\n")
        for day_num in range(self.days):
            current_date = self.start_date + timedelta(days=day_num)
            stats = self.simulate_day(current_date)
            self.daily_stats.append(stats)
            
            # Save to history
            self.revenue_history[stats.date] = {
                "revenue": stats.revenue,
                "cars": stats.cars,
                "avg_price": stats.avg_price,
                "occupancy": stats.occupancy,
                "weather": stats.weather,
                "is_weekend": stats.is_weekend,
                "is_holiday": stats.is_holiday
            }
            
            # Progress indicator
            if (day_num + 1) % 7 == 0 or day_num + 1 == self.days:
                week_num = (day_num + 1) // 7
                print(f"  ✓ Week {week_num} complete")
        
        print("\n✅ Simulation complete!\n")
        self._save_history()
    
    def display_daily_report(self) -> None:
        """Display day-by-day report"""
        
        print("="*100)
        print("📊 DAY-BY-DAY REVENUE REPORT")
        print("="*100)
        print()
        
        print(f"{'Date':<12} {'Day':<10} {'Revenue':<12} {'Cars':<8} {'Avg Price':<10} "
              f"{'Occup %':<9} {'Weather':<10} {'Status':<15}")
        print("-"*100)
        
        for stats in self.daily_stats:
            status = ""
            if stats.is_holiday:
                status = "🎉 HOLIDAY"
            elif stats.is_weekend:
                status = "📅 WEEKEND"
            else:
                status = "📆 WEEKDAY"
            
            weather_icon = {
                "Sunny": "☀️",
                "Rainy": "🌧️",
                "Cloudy": "☁️",
                "Snowy": "❄️",
                "Foggy": "🌫️"
            }.get(stats.weather, "❓")
            
            print(f"{stats.date:<12} {stats.day_name:<10} ${stats.revenue:<11.2f} "
                  f"{stats.cars:<8} ${stats.avg_price:<9.2f} {stats.occupancy:<8.1f}% "
                  f"{weather_icon} {stats.weather:<7} {status:<15}")
        
        print("-"*100)
        print()
    
    def display_summary(self) -> None:
        """Display comprehensive summary"""
        
        # Calculate metrics
        total_revenue = sum(s.revenue for s in self.daily_stats)
        total_cars = sum(s.cars for s in self.daily_stats)
        avg_daily_revenue = total_revenue / len(self.daily_stats) if self.daily_stats else 0
        avg_cars_per_day = total_cars / len(self.daily_stats) if self.daily_stats else 0
        revenue_per_car = total_revenue / total_cars if total_cars > 0 else 0
        
        # Weekly breakdown
        weekly_stats = []
        for week in range(self.weeks):
            week_start = week * 7
            week_end = min((week + 1) * 7, len(self.daily_stats))
            week_days = self.daily_stats[week_start:week_end]
            
            week_revenue = sum(s.revenue for s in week_days)
            week_cars = sum(s.cars for s in week_days)
            week_data = {
                "week": week + 1,
                "revenue": week_revenue,
                "cars": week_cars,
                "days": len(week_days)
            }
            weekly_stats.append(week_data)
        
        # Weather breakdown
        weather_stats = {}
        for stats in self.daily_stats:
            if stats.weather not in weather_stats:
                weather_stats[stats.weather] = {"revenue": 0, "count": 0}
            weather_stats[stats.weather]["revenue"] += stats.revenue
            weather_stats[stats.weather]["count"] += 1
        
        # Weekday vs Weekend
        weekday_revenue = sum(s.revenue for s in self.daily_stats if not s.is_weekend)
        weekend_revenue = sum(s.revenue for s in self.daily_stats if s.is_weekend)
        weekday_count = sum(1 for s in self.daily_stats if not s.is_weekend)
        weekend_count = sum(1 for s in self.daily_stats if s.is_weekend)
        
        # Display summary
        print("="*80)
        print("💰 OVERALL SUMMARY")
        print("="*80)
        print()
        print(f"Total Revenue:        ${total_revenue:,.2f}")
        print(f"Total Cars:           {total_cars:,}")
        print(f"Average Daily Revenue: ${avg_daily_revenue:,.2f}")
        print(f"Average Cars/Day:      {avg_cars_per_day:.0f}")
        print(f"Revenue per Car:      ${revenue_per_car:.2f}")
        print()
        
        print("="*80)
        print("📅 WEEKLY BREAKDOWN")
        print("="*80)
        print()
        
        print(f"{'Week':<8} {'Revenue':<15} {'Cars':<10} {'Avg/Day':<15}")
        print("-"*50)
        
        for week_data in weekly_stats:
            avg_week_revenue = week_data["revenue"] / week_data["days"]
            print(f"Week {week_data['week']:<3} ${week_data['revenue']:>12,.2f}  "
                  f"{week_data['cars']:>8}  ${avg_week_revenue:>12,.2f}")
        
        print("-"*50)
        
        # Best and worst weeks
        best_week = max(weekly_stats, key=lambda x: x["revenue"])
        worst_week = min(weekly_stats, key=lambda x: x["revenue"])
        
        print(f"\n📈 Best Week:  Week {best_week['week']} - ${best_week['revenue']:,.2f}")
        print(f"📉 Worst Week: Week {worst_week['week']} - ${worst_week['revenue']:,.2f}")
        print()
        
        print("="*80)
        print("🌦️  WEATHER IMPACT ANALYSIS")
        print("="*80)
        print()
        
        print(f"{'Weather':<12} {'Days':<8} {'Total Revenue':<18} {'Avg/Day':<15}")
        print("-"*50)
        
        for weather, data in sorted(weather_stats.items(), key=lambda x: x[1]["revenue"], reverse=True):
            avg_weather_revenue = data["revenue"] / data["count"]
            print(f"{weather:<12} {data['count']:<8} ${data['revenue']:>15,.2f}  "
                  f"${avg_weather_revenue:>12,.2f}")
        
        print()
        
        print("="*80)
        print("📊 WEEKDAY VS WEEKEND ANALYSIS")
        print("="*80)
        print()
        
        avg_weekday = weekday_revenue / weekday_count if weekday_count > 0 else 0
        avg_weekend = weekend_revenue / weekend_count if weekend_count > 0 else 0
        
        print(f"Weekdays:      {weekday_count} days - ${weekday_revenue:,.2f} total - "
              f"${avg_weekday:,.2f} avg/day")
        print(f"Weekends:      {weekend_count} days - ${weekend_revenue:,.2f} total - "
              f"${avg_weekend:,.2f} avg/day")
        print()
        
        if avg_weekend > avg_weekday:
            diff = ((avg_weekend - avg_weekday) / avg_weekday) * 100
            print(f"💡 Weekends earn {diff:.1f}% MORE than weekdays!")
        else:
            diff = ((avg_weekday - avg_weekend) / avg_weekend) * 100
            print(f"💡 Weekdays earn {diff:.1f}% MORE than weekends!")
        
        print()
        
        # Best and worst days
        best_day = max(self.daily_stats, key=lambda x: x.revenue)
        worst_day = min(self.daily_stats, key=lambda x: x.revenue)
        
        print("="*80)
        print("🏆 BEST AND WORST DAYS")
        print("="*80)
        print()
        
        print(f"🥇 Best Day:  {best_day.date} ({best_day.day_name})")
        print(f"   Revenue: ${best_day.revenue:,.2f}")
        print(f"   Cars: {best_day.cars}, Weather: {best_day.weather}")
        print()
        
        print(f"📉 Worst Day: {worst_day.date} ({worst_day.day_name})")
        print(f"   Revenue: ${worst_day.revenue:,.2f}")
        print(f"   Cars: {worst_day.cars}, Weather: {worst_day.weather}")
        print()
        
        print("="*80)
        print("✅ Simulation Results Saved to: revenue_history_automated.json")
        print("="*80)
        print()


def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("🚗 AUTOMATED PARKING LOT SIMULATOR - MULTI-WEEK RUNNER")
    print("="*80)
    print()
    
    # Get weeks input
    while True:
        try:
            weeks_input = input("How many weeks would you like to simulate? (1-12): ").strip()
            weeks = int(weeks_input)
            
            if 1 <= weeks <= 12:
                break
            else:
                print("❌ Please enter a number between 1 and 12")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print()
    
    # Create and run simulator
    simulator = AutomatedParkingSimulator(weeks=weeks)
    
    print("Starting simulation...")
    simulator.run_simulation()
    
    # Display results
    simulator.display_daily_report()
    simulator.display_summary()
    
    print("\n✨ Simulation complete! Check the detailed report above. ✨\n")


if __name__ == "__main__":
    main()
