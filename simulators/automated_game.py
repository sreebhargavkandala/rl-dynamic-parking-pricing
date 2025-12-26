"""
Automated Multi-Week Parking Lot Simulator Game
===============================================
Actual interactive game that runs for multiple weeks!
- Real cars parking/leaving
- Dynamic pricing updating
- Weather and weekend effects
- Revenue accumulating
- Week-by-week progression
- All with actual game mechanics and animations
"""

import pygame
import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Car:
    """Parked car"""
    car_id: int
    entry_time: datetime
    duration_hours: float
    assigned_price: float
    x: int
    y: int
    status: str = "parked"


class WeatherSimulator:
    """Simulates weather"""
    def __init__(self):
        self.weather_types = ["Sunny", "Rainy", "Cloudy", "Snowy", "Foggy"]
        self.weather_effects = {
            "Sunny": 1.0, "Rainy": 0.85, "Cloudy": 0.95, "Snowy": 0.7, "Foggy": 0.9
        }
        self.current_weather = "Sunny"
        self.weather_emoji = {"Sunny": "☀️", "Rainy": "🌧️", "Cloudy": "☁️", "Snowy": "❄️", "Foggy": "🌫️"}
    
    def get_weather_effect(self) -> float:
        return self.weather_effects[self.current_weather]
    
    def update_weather(self):
        self.current_weather = random.choices(
            self.weather_types,
            weights=[0.4, 0.2, 0.2, 0.1, 0.1]
        )[0]


class PricingAgent:
    """Dynamic pricing"""
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 25.0
        self.min_price = 1.5
    
    def calculate_price(self, occupancy: float, hour: int, 
                       weather_effect: float, is_weekend: bool) -> float:
        occupancy_factor = occupancy ** 2
        
        if is_weekend:
            peak_hours = [10, 11, 12, 13, 14, 18, 19, 20, 21]
        else:
            peak_hours = [8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
        
        peak_multiplier = 1.4 if hour in peak_hours else 1.0
        weekend_multiplier = 1.15 if is_weekend else 1.0
        
        price = self.base_price + (occupancy_factor * (self.max_price - self.base_price))
        price = price * peak_multiplier * weekend_multiplier * weather_effect
        
        return max(self.min_price, min(self.max_price, round(price, 2)))


class ParkingLot:
    """Parking lot management"""
    def __init__(self, rows=5, cols=8):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.cars = {}
        self.next_car_id = 1000
    
    def get_occupancy(self) -> float:
        return len(self.cars) / (self.rows * self.cols)
    
    def get_available_spots(self) -> List:
        available = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] is None:
                    available.append((r, c))
        return available
    
    def can_add_car(self) -> bool:
        return len(self.get_available_spots()) > 0
    
    def add_car(self, car: Car) -> bool:
        if not self.can_add_car():
            return False
        spot = random.choice(self.get_available_spots())
        car.x, car.y = spot
        self.grid[spot[0]][spot[1]] = car.car_id
        self.cars[car.car_id] = car
        return True
    
    def remove_car(self, car_id: int) -> bool:
        if car_id not in self.cars:
            return False
        car = self.cars[car_id]
        self.grid[car.x][car.y] = None
        del self.cars[car_id]
        return True
    
    def get_car_at(self, x: int, y: int) -> Optional[Car]:
        if 0 <= x < self.rows and 0 <= y < self.cols:
            car_id = self.grid[x][y]
            if car_id and car_id in self.cars:
                return self.cars[car_id]
        return None


class MultiWeekSimulator:
    """Main game with multi-week support"""
    
    def __init__(self, total_weeks: int = 4):
        pygame.init()
        
        self.width = 1400
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Automated Multi-Week Parking Lot Game")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # Game state
        self.parking_lot = ParkingLot(rows=5, cols=8)
        self.weather_sim = WeatherSimulator()
        self.pricing_agent = PricingAgent()
        self.running = True
        self.hovered_car = None
        self.mouse_pos = (0, 0)
        self.message = ""
        self.message_time = 0
        
        # Simulation tracking
        self.total_weeks = total_weeks
        self.total_days = total_weeks * 7
        self.current_day = 0
        self.current_date = datetime.now()
        self.simulation_time = datetime.now().replace(hour=6)
        
        # Weekly and daily tracking
        self.daily_revenues = []
        self.weekly_revenues = []
        self.week_car_count = 0
        self.day_car_count = 0
        self.day_revenue = 0.0
        self.week_revenue = 0.0
        self.total_revenue = 0.0
        self.all_cars_count = 0
        
        # UI
        self.lot_start_x = 50
        self.lot_start_y = 100
        self.cell_size = 60
        
        # Colors
        self.colors = {
            "bg": (240, 240, 245),
            "grid": (200, 200, 200),
            "car": (100, 150, 255),
            "car_hover": (50, 100, 200),
            "empty": (220, 220, 220),
            "text": (30, 30, 30),
            "green": (76, 175, 80),
            "red": (244, 67, 54),
            "blue": (33, 150, 243),
            "orange": (255, 152, 0)
        }
        
        # Auto-add cars parameters
        self.auto_add_timer = 0
        self.auto_add_interval = 0.5  # seconds between auto-adds
        self.simulation_speed = 100  # Days per second
    
    def show_message(self, text: str, duration: float = 2.0):
        self.message = text
        self.message_time = duration
    
    def update_day(self):
        """Simulate one day"""
        # Update weather
        self.weather_sim.update_weather()
        
        # Simulate the day
        is_weekend = self.current_date.weekday() >= 5
        occupancy = self.parking_lot.get_occupancy()
        hour = self.simulation_time.hour
        
        # Calculate current price
        price = self.pricing_agent.calculate_price(
            occupancy, hour, self.weather_sim.get_weather_effect(), is_weekend
        )
        
        # Auto-add some cars based on hour
        if hour in [9, 12, 18]:  # Peak hours
            num_cars = random.randint(3, 6)
        elif hour in [8, 10, 11, 13, 17, 19, 20]:  # Semi-peak
            num_cars = random.randint(2, 4)
        else:
            num_cars = random.randint(0, 2)
        
        # Apply weather effect
        if self.weather_sim.current_weather in ["Snowy", "Rainy"]:
            num_cars = int(num_cars * 0.7)
        
        # Add cars
        for _ in range(num_cars):
            if self.parking_lot.can_add_car():
                duration = random.uniform(0.5, 3.0)
                car = Car(
                    car_id=self.parking_lot.next_car_id,
                    entry_time=self.simulation_time,
                    duration_hours=duration,
                    assigned_price=price,
                    x=0, y=0
                )
                self.parking_lot.next_car_id += 1
                
                if self.parking_lot.add_car(car):
                    self.day_revenue += price
                    self.day_car_count += 1
                    self.all_cars_count += 1
    
    def advance_hour(self, dt: float):
        """Advance simulation by hour"""
        self.simulation_time += timedelta(hours=1)
        
        # Check for cars leaving
        cars_to_remove = []
        for car_id, car in self.parking_lot.cars.items():
            age_minutes = (self.simulation_time - car.entry_time).total_seconds() / 60
            if age_minutes >= (car.duration_hours * 60):
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            self.parking_lot.remove_car(car_id)
        
        # Update day if hit midnight
        if self.simulation_time.hour == 0 and self.simulation_time.minute == 0:
            self.end_day()
    
    def end_day(self):
        """End current day and start new one"""
        # Save day data
        self.daily_revenues.append(self.day_revenue)
        self.week_revenue += self.day_revenue
        self.week_car_count += self.day_car_count
        self.total_revenue += self.day_revenue
        
        # Move to next day
        self.current_day += 1
        self.current_date += timedelta(days=1)
        
        # Check for week end
        if self.current_day % 7 == 0:
            self.weekly_revenues.append(self.week_revenue)
            week_num = self.current_day // 7
            self.show_message(f"✅ Week {week_num} Complete! Revenue: ${self.week_revenue:,.2f}")
            self.week_revenue = 0.0
            self.week_car_count = 0
        
        # Reset day
        self.day_revenue = 0.0
        self.day_car_count = 0
        self.parking_lot = ParkingLot(rows=5, cols=8)
        self.simulation_time = self.current_date.replace(hour=6)
    
    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.hovered_car = self.get_car_at_mouse()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.auto_add_interval = 0.1 if self.auto_add_interval == 0.5 else 0.5
    
    def get_car_at_mouse(self) -> Optional[Car]:
        """Get car under mouse"""
        x, y = self.mouse_pos
        if not (self.lot_start_x <= x <= self.lot_start_x + 480 and
                self.lot_start_y <= y <= self.lot_start_y + 300):
            return None
        
        grid_x = (x - self.lot_start_x) // self.cell_size
        grid_y = (y - self.lot_start_y) // self.cell_size
        
        return self.parking_lot.get_car_at(int(grid_x), int(grid_y))
    
    def update(self, dt: float):
        """Update game state"""
        if self.current_day >= self.total_days:
            self.running = False
            return
        
        # Update message timer
        if self.message_time > 0:
            self.message_time -= dt
        
        # Advance simulation time (30x speed)
        for _ in range(30):
            self.advance_hour(dt / 30)
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(self.colors["bg"])
        
        # Title with week info
        week_num = (self.current_day // 7) + 1
        day_name = self.current_date.strftime("%A")
        title = self.font_large.render(
            f"🚗 Multi-Week Parking Simulator | Week {week_num} | {self.current_date.strftime('%Y-%m-%d')} ({day_name})",
            True, self.colors["text"]
        )
        self.screen.blit(title, (20, 20))
        
        # Draw parking lot
        self.draw_parking_lot()
        
        # Draw info panel
        self.draw_info_panel()
        
        # Draw hover tooltip
        if self.hovered_car:
            self.draw_car_tooltip()
        
        # Draw message
        if self.message_time > 0:
            self.draw_message()
        
        # Draw progress
        self.draw_progress()
        
        pygame.display.flip()
    
    def draw_parking_lot(self):
        """Draw parking lot grid and cars"""
        # Grid
        for r in range(5):
            for c in range(8):
                x = self.lot_start_x + c * self.cell_size
                y = self.lot_start_y + r * self.cell_size
                
                pygame.draw.rect(self.screen, self.colors["empty"],
                               (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors["grid"],
                               (x, y, self.cell_size, self.cell_size), 2)
        
        # Cars
        for car in self.parking_lot.cars.values():
            x = self.lot_start_x + car.y * self.cell_size + 5
            y = self.lot_start_y + car.x * self.cell_size + 5
            size = self.cell_size - 10
            
            color = (self.colors["car_hover"] if car == self.hovered_car 
                    else self.colors["car"])
            
            pygame.draw.rect(self.screen, color, (x, y, size, size))
            pygame.draw.rect(self.screen, self.colors["text"], (x, y, size, size), 2)
            
            car_text = self.font_small.render(str(car.car_id), True, (255, 255, 255))
            text_rect = car_text.get_rect(center=(x + size//2, y + size//2))
            self.screen.blit(car_text, text_rect)
    
    def draw_info_panel(self):
        """Draw right panel with info"""
        x = self.lot_start_x + 520
        y = 120
        
        # Current day info
        info_lines = [
            f"📅 Day: {self.current_day + 1}/{self.total_days}",
            f"⏰ Time: {self.simulation_time.strftime('%H:%M')}",
            f"🌦️  {self.weather_sim.current_weather}",
            f"",
            f"📊 Occupancy: {len(self.parking_lot.cars)}/40 ({self.parking_lot.get_occupancy()*100:.1f}%)",
            f"💰 Today: ${self.day_revenue:,.2f}",
            f"🚗 Today: {self.day_car_count} cars",
            f"",
            f"📈 Week {(self.current_day // 7) + 1}: ${self.week_revenue:,.2f}",
            f"",
            f"💵 Total: ${self.total_revenue:,.2f}",
            f"🚗 Total: {self.all_cars_count} cars",
        ]
        
        for i, line in enumerate(info_lines):
            color = self.colors["green"] if "Total" in line else self.colors["text"]
            t = self.font_medium.render(line, True, color)
            self.screen.blit(t, (x, y + i * 25))
    
    def draw_car_tooltip(self):
        """Draw tooltip for hovered car"""
        car = self.hovered_car
        x, y = self.mouse_pos
        
        lines = [
            f"Car #{car.car_id}",
            f"Price: ${car.assigned_price:.2f}",
            f"Duration: {car.duration_hours:.1f}h",
        ]
        
        max_width = max(len(line) for line in lines) * 8
        tooltip_height = len(lines) * 22 + 10
        
        tooltip_x = min(x + 10, self.width - max_width - 10)
        tooltip_y = min(y + 10, self.height - tooltip_height - 10)
        
        pygame.draw.rect(self.screen, (50, 50, 50),
                        (tooltip_x, tooltip_y, max_width, tooltip_height))
        pygame.draw.rect(self.screen, self.colors["text"],
                        (tooltip_x, tooltip_y, max_width, tooltip_height), 1)
        
        for i, line in enumerate(lines):
            t = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(t, (tooltip_x + 5, tooltip_y + 5 + i * 22))
    
    def draw_message(self):
        """Draw temporary message"""
        t = self.font_medium.render(self.message, True, self.colors["green"])
        t_rect = t.get_rect(center=(self.width // 2, 60))
        
        bg_rect = t_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect)
        pygame.draw.rect(self.screen, self.colors["green"], bg_rect, 2)
        
        self.screen.blit(t, t_rect)
    
    def draw_progress(self):
        """Draw overall progress bar"""
        progress = self.current_day / self.total_days
        bar_width = 300
        bar_height = 20
        x = self.width - bar_width - 20
        y = self.height - 40
        
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.colors["green"], 
                        (x, y, bar_width * progress, bar_height))
        pygame.draw.rect(self.screen, self.colors["text"], 
                        (x, y, bar_width, bar_height), 2)
        
        progress_text = self.font_small.render(f"{progress*100:.0f}%", True, self.colors["text"])
        self.screen.blit(progress_text, (x + 10, y + 2))
    
    def run(self):
        """Main game loop"""
        print("🎮 Starting automated multi-week parking lot game...")
        print(f"   Running for {self.total_weeks} weeks ({self.total_days} days)")
        print("   Watch the game window for real-time updates!\n")
        
        while self.running and self.current_day < self.total_days:
            dt = self.clock.tick(60) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        # Display final summary
        print("\n" + "="*80)
        print("🎉 SIMULATION COMPLETE!")
        print("="*80)
        print(f"Total Revenue: ${self.total_revenue:,.2f}")
        print(f"Total Cars: {self.all_cars_count}")
        print(f"Days Simulated: {self.current_day}")
        print(f"Weeks Simulated: {len(self.weekly_revenues)}")
        if self.weekly_revenues:
            print(f"\nWeekly Breakdown:")
            for i, week_rev in enumerate(self.weekly_revenues):
                print(f"  Week {i+1}: ${week_rev:,.2f}")
        print("="*80 + "\n")
        
        pygame.quit()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚗 AUTOMATED MULTI-WEEK PARKING LOT GAME")
    print("="*80 + "\n")
    
    while True:
        try:
            weeks = int(input("How many weeks to simulate? (1-12): ").strip())
            if 1 <= weeks <= 12:
                break
            print("❌ Please enter 1-12")
        except:
            print("❌ Invalid input")
    
    game = MultiWeekSimulator(total_weeks=weeks)
    game.run()


if __name__ == "__main__":
    main()
