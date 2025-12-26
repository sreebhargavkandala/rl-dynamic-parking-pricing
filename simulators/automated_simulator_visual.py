"""
Automated Parking Lot Simulator - Visual Multi-Week Run with GUI
=================================================================

Shows complete week-by-week simulation with:
- Visual parking lot representation
- Real-time daily updates
- Weather effects display
- Dynamic pricing visualization
- Weekly and overall revenue tracking
- All in an interactive pygame window
"""

import pygame
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random
import json
from typing import Dict, List, Tuple


class WeatherSimulator:
    """Simulates weather effects"""
    
    def __init__(self):
        self.weather_types = ["Sunny", "Rainy", "Cloudy", "Snowy", "Foggy"]
        self.weather_effects = {
            "Sunny": 1.0,
            "Rainy": 0.85,
            "Cloudy": 0.95,
            "Snowy": 0.7,
            "Foggy": 0.9
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
        self.holidays = {
            "01-01": ("New Year", 0.6),
            "12-25": ("Christmas", 0.5),
            "07-04": ("Independence Day", 0.7),
            "11-27": ("Thanksgiving", 0.8),
        }
    
    def get_holiday_info(self, date: datetime) -> Tuple[bool, str, float]:
        """Check if day is holiday"""
        date_key = date.strftime("%m-%d")
        if date_key in self.holidays:
            name, effect = self.holidays[date_key]
            return True, name, effect
        return False, "", 1.0


class AdvancedPricingAgent:
    """Advanced pricing with weather, time, occupancy"""
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 25.0
        self.min_price = 1.5
    
    def calculate_price(self, occupancy: float, hour: int, 
                       weather_effect: float, is_weekend: bool, 
                       holiday_effect: float) -> float:
        """Calculate dynamic price"""
        occupancy_factor = occupancy ** 2
        
        if is_weekend:
            peak_hours = [10, 11, 12, 13, 14, 18, 19, 20, 21]
        else:
            peak_hours = [8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
        
        peak_multiplier = 1.4 if hour in peak_hours else 1.0
        weekend_multiplier = 1.15 if is_weekend else 1.0
        
        price = self.base_price + (occupancy_factor * (self.max_price - self.base_price))
        price = price * peak_multiplier * weekend_multiplier * weather_effect * holiday_effect
        
        price = max(self.min_price, min(self.max_price, price))
        return round(price, 2)


class VisualAutomatedSimulator:
    """Visual GUI simulator for multi-week runs"""
    
    def __init__(self, weeks: int = 4):
        pygame.init()
        
        self.width = 1600
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Automated Parking Lot Simulator - Multi-Week Run")
        
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # Simulation
        self.weeks = weeks
        self.days = weeks * 7
        self.weather_sim = WeatherSimulator()
        self.holiday_sim = HolidaySimulator()
        self.pricing_agent = AdvancedPricingAgent()
        self.start_date = datetime.now()
        
        # Data
        self.daily_data = []
        self.weekly_data = []
        self.current_day = 0
        self.running = True
        self.simulation_done = False
        self.paused = False
        self.speed = 1
        
        # Colors
        self.colors = {
            "bg": (240, 240, 245),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "green": (76, 175, 80),
            "red": (244, 67, 54),
            "blue": (33, 150, 243),
            "orange": (255, 152, 0),
            "text": (30, 30, 30),
            "light_gray": (200, 200, 200)
        }
    
    def simulate_day(self, day_num: int) -> Dict:
        """Simulate a single day"""
        date = self.start_date + timedelta(days=day_num)
        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")
        is_weekend = date.weekday() >= 5
        
        weather, weather_effect = self.weather_sim.get_weather_and_effect(date)
        is_holiday, holiday_name, holiday_effect = self.holiday_sim.get_holiday_info(date)
        
        total_revenue = 0.0
        total_cars = 0
        prices = []
        occupancies = []
        
        for hour in range(24):
            if is_weekend:
                base_occupancy = (
                    0.3 * np.sin((hour - 11) * np.pi / 24) + 
                    0.2 * np.sin((hour - 18) * np.pi / 24) + 0.5
                )
            else:
                base_occupancy = (
                    0.35 * np.sin((hour - 9) * np.pi / 24) + 
                    0.25 * np.sin((hour - 17) * np.pi / 24) + 0.4
                )
            
            occupancy = max(0.0, min(1.0, base_occupancy * weather_effect * holiday_effect))
            occupancies.append(occupancy)
            
            price = self.pricing_agent.calculate_price(
                occupancy, hour, weather_effect, is_weekend, holiday_effect
            )
            prices.append(price)
            
            cars_this_hour = max(0, int(
                (5 + occupancy * 8) * weather_effect * (1 if hour in [9, 12, 18] else 0.7)
            ))
            
            if random.random() > (1 - weather_effect):
                cars_this_hour += 1
            
            total_cars += cars_this_hour
            total_revenue += cars_this_hour * price
        
        return {
            "date": date_str,
            "day": day_num,
            "day_name": day_name,
            "revenue": round(total_revenue, 2),
            "cars": total_cars,
            "avg_price": round(np.mean(prices), 2),
            "occupancy": round(np.mean(occupancies) * 100, 1),
            "weather": weather,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name
        }
    
    def run_simulation(self) -> None:
        """Run simulation loop"""
        
        while self.running and self.current_day < self.days:
            dt = self.clock.tick(60) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.speed = min(10, self.speed + 1)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(1, self.speed - 1)
            
            # Update simulation
            if not self.paused and not self.simulation_done:
                # Simulate day
                day_data = self.simulate_day(self.current_day)
                self.daily_data.append(day_data)
                
                # Update weekly data
                week_num = self.current_day // 7
                if week_num >= len(self.weekly_data):
                    self.weekly_data.append({
                        "week": week_num + 1,
                        "revenue": 0,
                        "cars": 0,
                        "days": []
                    })
                
                self.weekly_data[week_num]["revenue"] += day_data["revenue"]
                self.weekly_data[week_num]["cars"] += day_data["cars"]
                self.weekly_data[week_num]["days"].append(day_data)
                
                self.current_day += 1
                
                if self.current_day >= self.days:
                    self.simulation_done = True
            
            # Draw
            self.draw()
        
        # Save results
        self.save_results()
    
    def draw(self) -> None:
        """Draw the simulator"""
        self.screen.fill(self.colors["bg"])
        
        # Title
        title = self.font_title.render("🚗 AUTOMATED MULTI-WEEK PARKING SIMULATOR", True, self.colors["text"])
        self.screen.blit(title, (20, 20))
        
        # Progress bar
        self.draw_progress()
        
        # Current day info
        if self.daily_data:
            self.draw_current_day_info()
        
        # Weekly summary
        self.draw_weekly_summary()
        
        # Overall stats
        self.draw_overall_stats()
        
        # Controls
        self.draw_controls()
        
        pygame.display.flip()
    
    def draw_progress(self) -> None:
        """Draw simulation progress"""
        progress = self.current_day / self.days if self.days > 0 else 0
        bar_width = 500
        bar_height = 30
        x, y = 20, 70
        
        # Background
        pygame.draw.rect(self.screen, self.colors["light_gray"], (x, y, bar_width, bar_height))
        
        # Progress
        fill_width = bar_width * progress
        pygame.draw.rect(self.screen, self.colors["green"], (x, y, fill_width, bar_height))
        
        # Text
        progress_text = self.font_medium.render(
            f"Progress: Day {self.current_day}/{self.days} ({progress*100:.1f}%)",
            True, self.colors["text"]
        )
        self.screen.blit(progress_text, (x + 10, y + 5))
    
    def draw_current_day_info(self) -> None:
        """Draw current day information"""
        if not self.daily_data:
            return
        
        today = self.daily_data[-1]
        x, y = 20, 120
        
        # Box background
        pygame.draw.rect(self.screen, (255, 255, 255), (x - 5, y - 5, 350, 200), 0)
        pygame.draw.rect(self.screen, self.colors["blue"], (x - 5, y - 5, 350, 200), 2)
        
        # Content
        texts = [
            f"📅 Current Day: {today['date']}",
            f"📆 {today['day_name']}" + (" (🏖️ Weekend)" if today['is_weekend'] else " (📆 Weekday)"),
            f"🌦️  Weather: {today['weather']}",
            f"💰 Revenue: ${today['revenue']:,.2f}",
            f"🚗 Cars: {today['cars']}",
            f"💵 Avg Price: ${today['avg_price']:.2f}",
            f"📊 Occupancy: {today['occupancy']:.1f}%"
        ]
        
        if today['is_holiday']:
            texts.insert(2, f"🎉 Holiday: {today['holiday_name']}")
        
        for i, text in enumerate(texts):
            t = self.font_small.render(text, True, self.colors["text"])
            self.screen.blit(t, (x + 10, y + 10 + i * 20))
    
    def draw_weekly_summary(self) -> None:
        """Draw weekly summary"""
        x, y = 400, 120
        
        pygame.draw.rect(self.screen, (255, 255, 255), (x - 5, y - 5, 450, 200), 0)
        pygame.draw.rect(self.screen, self.colors["orange"], (x - 5, y - 5, 450, 200), 2)
        
        title = self.font_large.render("📅 WEEKLY SUMMARY", True, self.colors["text"])
        self.screen.blit(title, (x + 10, y))
        
        y_offset = y + 35
        for week_data in self.weekly_data:
            week_text = self.font_small.render(
                f"Week {week_data['week']}: ${week_data['revenue']:,.2f} | "
                f"{week_data['cars']} cars | {len(week_data['days'])}/7 days simulated",
                True, self.colors["text"]
            )
            self.screen.blit(week_text, (x + 10, y_offset))
            y_offset += 25
    
    def draw_overall_stats(self) -> None:
        """Draw overall statistics"""
        x, y = 900, 120
        
        pygame.draw.rect(self.screen, (255, 255, 255), (x - 5, y - 5, 650, 200), 0)
        pygame.draw.rect(self.screen, self.colors["green"], (x - 5, y - 5, 650, 200), 2)
        
        title = self.font_large.render("💰 OVERALL STATISTICS", True, self.colors["text"])
        self.screen.blit(title, (x + 10, y))
        
        if self.daily_data:
            total_revenue = sum(d["revenue"] for d in self.daily_data)
            total_cars = sum(d["cars"] for d in self.daily_data)
            avg_daily = total_revenue / len(self.daily_data) if self.daily_data else 0
            revenue_per_car = total_revenue / total_cars if total_cars > 0 else 0
            
            stats = [
                f"Total Revenue: ${total_revenue:,.2f}",
                f"Total Cars: {total_cars}",
                f"Avg Daily Revenue: ${avg_daily:,.2f}",
                f"Revenue per Car: ${revenue_per_car:.2f}",
                f"Days Simulated: {len(self.daily_data)}/{self.days}"
            ]
            
            y_offset = y + 35
            for stat in stats:
                t = self.font_small.render(stat, True, self.colors["text"])
                self.screen.blit(t, (x + 10, y_offset))
                y_offset += 25
    
    def draw_controls(self) -> None:
        """Draw control instructions"""
        x, y = 20, self.height - 100
        
        controls = [
            "CONTROLS:",
            "SPACE: Pause/Resume",
            "↑/↓: Change Speed",
            "ESC: Exit"
        ]
        
        for i, ctrl in enumerate(controls):
            color = self.colors["blue"] if i == 0 else self.colors["text"]
            t = self.font_small.render(ctrl, True, color)
            self.screen.blit(t, (x, y + i * 20))
        
        # Status
        status = "⏸️  PAUSED" if self.paused else "▶️  RUNNING"
        status_text = self.font_medium.render(f"{status} (Speed: {self.speed}x)", True, 
                                            self.colors["green"] if not self.paused else self.colors["red"])
        self.screen.blit(status_text, (x, y + 100))
    
    def save_results(self) -> None:
        """Save simulation results"""
        results = {
            "simulation_date": datetime.now().isoformat(),
            "weeks": self.weeks,
            "daily_data": self.daily_data,
            "weekly_summary": self.weekly_data
        }
        
        with open("simulation_results_visual.json", 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚗 VISUAL AUTOMATED PARKING LOT SIMULATOR")
    print("="*80 + "\n")
    
    while True:
        try:
            weeks_input = input("How many weeks to simulate? (1-12): ").strip()
            weeks = int(weeks_input)
            if 1 <= weeks <= 12:
                break
            print("❌ Please enter 1-12")
        except ValueError:
            print("❌ Invalid input")
    
    print(f"\n🚀 Starting {weeks}-week simulation in visual window...\n")
    
    simulator = VisualAutomatedSimulator(weeks=weeks)
    simulator.run_simulation()
    
    if simulator.running:
        print("\n✅ Simulation complete! Results saved to simulation_results_visual.json")
        print("Close the window to exit.\n")
        
        # Keep window open
        while simulator.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    simulator.running = False
            simulator.draw()
            simulator.clock.tick(60)
    
    pygame.quit()
    print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
