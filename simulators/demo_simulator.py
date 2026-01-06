 

import pygame
import json
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os

pygame.init()
 
class Car:
    """Represents a parked car"""
    def __init__(self, car_id, entry_time, duration_hours, assigned_price):
        self.car_id = car_id
        self.entry_time = entry_time
        self.duration_hours = duration_hours
        self.assigned_price = assigned_price
        self.grid_x = None
        self.grid_y = None


class ParkingLot:
    """Manages the parking lot grid"""
    def __init__(self, rows=5, cols=8):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.cars = {}
        self.car_counter = 1000
    
    def get_occupancy(self):
        return len(self.cars) / (self.rows * self.cols)
    
    def add_car(self, car):
        available = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                     if self.grid[r][c] is None]
        if not available:
            return False
        
        r, c = random.choice(available)
        car.grid_x, car.grid_y = r, c
        self.grid[r][c] = car.car_id
        self.cars[car.car_id] = car
        return True
    
    def remove_car(self, car_id):
        if car_id in self.cars:
            car = self.cars[car_id]
            self.grid[car.grid_x][car.grid_y] = None
            del self.cars[car_id]
            return True
        return False
    
    def get_free_spaces(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols) 
                   if self.grid[r][c] is None)


class DynamicPricingEngine:
    """THE CORE ALGORITHM - Shows pricing calculation step by step"""
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 25.0
        self.min_price = 1.5
        
        # Weather effects
        self.weather_effects = {
            "Sunny": 1.0,
            "Cloudy": 0.95,
            "Rainy": 0.85,
            "Snowy": 0.70,
            "Foggy": 0.90
        }
        
        # Peak hour definitions
        self.peak_hours = [8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
        self.semi_peak = [7, 11, 15, 16, 21]
    
    def calculate_price_breakdown(self, occupancy, hour, is_weekend, weather):
        """
        Calculate price with detailed breakdown
        Returns: (final_price, breakdown_dict)
        """
        
        breakdown = {}
        
        # FACTOR 1: Base Price
        base = self.base_price
        breakdown['base_price'] = base
        
        # FACTOR 2: Occupancy (Quadratic - MAIN DRIVER)
        occupancy_factor = occupancy ** 2
        occupancy_addition = occupancy_factor * (self.max_price - self.base_price)
        breakdown['occupancy_factor'] = occupancy
        breakdown['occupancy_boost'] = occupancy_addition
        
        # FACTOR 3: Peak Hour Multiplier
        if hour in self.peak_hours:
            peak_mult = 1.4
            peak_label = "PEAK"
        elif hour in self.semi_peak:
            peak_mult = 1.2
            peak_label = "SEMI-PEAK"
        else:
            peak_mult = 0.85
            peak_label = "OFF-PEAK"
        breakdown['peak_multiplier'] = peak_mult
        breakdown['peak_label'] = peak_label
        
        # FACTOR 4: Weekend Multiplier
        weekend_mult = 1.15 if is_weekend else 1.0
        breakdown['weekend_multiplier'] = weekend_mult
        breakdown['is_weekend'] = is_weekend
        
        # FACTOR 5: Weather Effect
        weather_effect = self.weather_effects.get(weather, 1.0)
        breakdown['weather_effect'] = weather_effect
        breakdown['weather'] = weather
        
        # FINAL CALCULATION
        price = base + occupancy_addition
        price = price * peak_mult * weekend_mult * weather_effect
        final_price = max(self.min_price, min(self.max_price, round(price, 2)))
        
        breakdown['final_price'] = final_price
        breakdown['price_change'] = final_price - base
        
        return final_price, breakdown


class WeatherManager:
    """Manages weather changes"""
    def __init__(self):
        self.current_weather = "Sunny"
        self.weather_list = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
        self.weather_emoji = {
            "Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️", 
            "Snowy": "❄️", "Foggy": "🌫️"
        }
    
    def update(self):
        self.current_weather = random.choices(
            self.weather_list,
            weights=[0.4, 0.25, 0.15, 0.1, 0.1]
        )[0]


class DemoSimulator:
    """Main demo simulator"""
    
    def __init__(self):
        self.width = 1600
        self.height = 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🅿️ PARKING LOT DYNAMIC PRICING DEMO - Faculty Presentation")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Fonts
        self.font_title = pygame.font.Font(None, 32)
        self.font_header = pygame.font.Font(None, 26)
        self.font_normal = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.font_tiny = pygame.font.Font(None, 14)
        
        # Components
        self.parking_lot = ParkingLot(rows=5, cols=8)
        self.pricing_engine = DynamicPricingEngine()
        self.weather_manager = WeatherManager()
        
        # Simulation state
        self.current_time = datetime.now().replace(hour=6, minute=0, second=0)
        self.session_start = self.current_time
        self.simulation_day = 0
        
        # Tracking
        self.all_prices_history = []
        self.hourly_revenue = defaultdict(float)
        self.hourly_cars = defaultdict(int)
        self.total_revenue = 0.0
        self.total_cars_served = 0
        
        # UI State
        self.hovered_car = None
        self.mouse_pos = (0, 0)
        self.last_add_time = 0
        self.add_cars_interval = 0.3
        self.simulation_speed = 30  # Hours per second
        
        # Colors
        self.colors = {
            'bg': (15, 15, 25),
            'panel_bg': (25, 25, 40),
            'grid_empty': (50, 50, 70),
            'grid_occupied': (70, 100, 150),
            'car': (100, 200, 255),
            'car_hover': (150, 220, 255),
            'text': (240, 240, 250),
            'text_dim': (150, 150, 170),
            'accent': (100, 200, 100),
            'warning': (255, 150, 100),
            'peak': (255, 100, 100),
            'grid': (60, 60, 80),
        }
        
        self.lot_x = 30
        self.lot_y = 120
        self.cell_size = 50
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.hovered_car = self.get_car_under_mouse()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def get_car_under_mouse(self):
        x, y = self.mouse_pos
        if not (self.lot_x <= x <= self.lot_x + 400 and
                self.lot_y <= y <= self.lot_y + 250):
            return None
        
        grid_x = (x - self.lot_x) // self.cell_size
        grid_y = (y - self.lot_y) // self.cell_size
        
        if 0 <= grid_x < 8 and 0 <= grid_y < 5:
            car_id = self.parking_lot.grid[int(grid_y)][int(grid_x)]
            if car_id:
                return self.parking_lot.cars.get(car_id)
        return None
    
    def update(self, dt):
        if self.paused or not self.running:
            return
        
        # Auto-add cars at intervals
        self.last_add_time += dt
        if self.last_add_time >= self.add_cars_interval:
            self.add_new_cars()
            self.last_add_time = 0
        
        # Advance time
        self.current_time += timedelta(minutes=self.simulation_speed * dt * 60)
        
        # Remove expired cars
        self.remove_expired_cars()
        
        # Update weather every hour
        if self.current_time.minute == 0 and int(self.current_time.second) == 0:
            self.weather_manager.update()
    
    def add_new_cars(self):
        """Add cars based on time and weather"""
        hour = self.current_time.hour
        occupancy = self.parking_lot.get_occupancy()
        
        # Determine how many cars to add based on hour
        if hour in [8, 9, 10, 17, 18, 19]:  # Peak hours
            num_cars = random.randint(3, 5)
        elif hour in [11, 12, 13, 14, 20, 21]:  # Medium
            num_cars = random.randint(2, 3)
        else:  # Off-peak
            num_cars = random.randint(0, 2)
        
        # Weather effect
        if self.weather_manager.current_weather in ["Snowy", "Rainy"]:
            num_cars = int(num_cars * 0.7)
        elif self.weather_manager.current_weather == "Sunny":
            num_cars = int(num_cars * 1.1)
        
        # Don't overfill
        free_spaces = self.parking_lot.get_free_spaces()
        num_cars = min(num_cars, free_spaces)
        
        # Add cars
        is_weekend = self.current_time.weekday() >= 5
        hour_val = self.current_time.hour
        
        for _ in range(num_cars):
            # Calculate price for this car
            occupancy = self.parking_lot.get_occupancy()
            price, breakdown = self.pricing_engine.calculate_price_breakdown(
                occupancy, hour_val, is_weekend, self.weather_manager.current_weather
            )
            
            # Create and add car
            car = Car(
                car_id=self.parking_lot.car_counter,
                entry_time=self.current_time,
                duration_hours=random.uniform(0.5, 3.0),
                assigned_price=price
            )
            self.parking_lot.car_counter += 1
            
            if self.parking_lot.add_car(car):
                self.total_revenue += price
                self.total_cars_served += 1
                self.hourly_revenue[hour_val] += price
                self.hourly_cars[hour_val] += 1
                self.all_prices_history.append(price)
    
    def remove_expired_cars(self):
        """Remove cars that have overstayed"""
        cars_to_remove = []
        for car_id, car in self.parking_lot.cars.items():
            age_hours = (self.current_time - car.entry_time).total_seconds() / 3600
            if age_hours >= car.duration_hours:
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            self.parking_lot.remove_car(car_id)
    
    def draw(self):
        """Main draw function"""
        self.screen.fill(self.colors['bg'])
        
        # Title
        title = self.font_title.render(
            "🅿️ DYNAMIC PARKING PRICING ALGORITHM DEMO",
            True, self.colors['accent']
        )
        self.screen.blit(title, (20, 20))
        
        # Left side: Parking Lot
        self.draw_parking_lot()
        
        # Right side: Pricing Breakdown
        self.draw_pricing_breakdown()
        
        # Bottom: Statistics
        self.draw_statistics()
        
        # Hover tooltip
        if self.hovered_car:
            self.draw_car_tooltip()
        
        # Pause indicator
        if self.paused:
            pause_text = self.font_header.render("⏸ PAUSED (SPACE to resume)", True, (255, 100, 100))
            self.screen.blit(pause_text, (self.width - 350, 20))
        
        pygame.display.flip()
    
    def draw_parking_lot(self):
        """Draw parking lot grid"""
        # Title
        title = self.font_header.render("PARKING LOT (5x8 Grid)", True, self.colors['text'])
        self.screen.blit(title, (self.lot_x, self.lot_y - 35))
        
        # Grid
        for row in range(5):
            for col in range(8):
                x = self.lot_x + col * self.cell_size
                y = self.lot_y + row * self.cell_size
                
                car_id = self.parking_lot.grid[row][col]
                if car_id:
                    color = self.colors['car_hover'] if (car_id == self.hovered_car.car_id if self.hovered_car else False) else self.colors['car']
                    pygame.draw.rect(self.screen, color, (x, y, self.cell_size-2, self.cell_size-2))
                    pygame.draw.rect(self.screen, self.colors['text'], (x, y, self.cell_size-2, self.cell_size-2), 1)
                else:
                    pygame.draw.rect(self.screen, self.colors['grid_empty'], (x, y, self.cell_size-2, self.cell_size-2))
                    pygame.draw.rect(self.screen, self.colors['grid'], (x, y, self.cell_size-2, self.cell_size-2), 1)
        
        # Occupancy info
        occupancy = self.parking_lot.get_occupancy()
        occ_text = self.font_normal.render(
            f"Occupancy: {len(self.parking_lot.cars)}/40 ({occupancy*100:.1f}%)",
            True, self.colors['accent']
        )
        self.screen.blit(occ_text, (self.lot_x, self.lot_y + 270))
    
    def draw_pricing_breakdown(self):
        """Draw pricing algorithm breakdown"""
        x = 500
        y = 120
        
        # Current time and weather
        time_str = self.current_time.strftime("%H:%M")
        day_str = self.current_time.strftime("%A")
        is_weekend = self.current_time.weekday() >= 5
        
        pygame.draw.line(self.screen, self.colors['grid'], (x-10, y-50), (self.width-20, y-50), 1)
        
        header = self.font_header.render("PRICING ALGORITHM BREAKDOWN", True, self.colors['accent'])
        self.screen.blit(header, (x, y))
        
        y += 40
        
        # Current conditions
        cond_texts = [
            f"⏰ Time: {time_str} ({day_str})",
            f"🌦️  Weather: {self.weather_manager.current_weather}",
            f"📍 Occupancy: {self.parking_lot.get_occupancy()*100:.1f}%",
        ]
        
        for text in cond_texts:
            t = self.font_normal.render(text, True, self.colors['text_dim'])
            self.screen.blit(t, (x, y))
            y += 25
        
        y += 10
        
        # Pricing factors
        occupancy = self.parking_lot.get_occupancy()
        hour = self.current_time.hour
        is_weekend = self.current_time.weekday() >= 5
        weather = self.weather_manager.current_weather
        
        price, breakdown = self.pricing_engine.calculate_price_breakdown(
            occupancy, hour, is_weekend, weather
        )
        
        # Draw each factor
        factors = [
            ("BASE PRICE", f"${breakdown['base_price']:.2f}", self.colors['text']),
            ("+ Occupancy Factor", f"(Occ² × Range) = ${breakdown['occupancy_boost']:.2f}", 
             (255, 200, 100) if occupancy > 0.7 else self.colors['text_dim']),
            ("× Peak Multiplier", f"{breakdown['peak_label']}: {breakdown['peak_multiplier']:.2f}x", 
             (255, 100, 100) if breakdown['peak_label'] == "PEAK" else self.colors['text_dim']),
            ("× Weekend Multiplier", f"{breakdown['weekend_multiplier']:.2f}x" + (" (WEEKEND)" if is_weekend else ""), 
             (255, 200, 100) if is_weekend else self.colors['text_dim']),
            ("× Weather Effect", f"{weather}: {breakdown['weather_effect']:.2f}x", 
             self.colors['text_dim']),
        ]
        
        for label, value, color in factors:
            label_t = self.font_normal.render(label, True, color)
            value_t = self.font_normal.render(value, True, color)
            self.screen.blit(label_t, (x, y))
            self.screen.blit(value_t, (x + 280, y))
            y += 28
        
        y += 10
        pygame.draw.line(self.screen, self.colors['grid'], (x, y), (x + 400, y), 2)
        y += 15
        
        # Final price
        final_text = self.font_header.render(
            f"FINAL PRICE: ${breakdown['final_price']:.2f}",
            True, (100, 255, 150)
        )
        self.screen.blit(final_text, (x, y))
        
        if breakdown['price_change'] > 0:
            change_text = self.font_small.render(
                f"(+${breakdown['price_change']:.2f} from base)",
                True, (100, 255, 150)
            )
            self.screen.blit(change_text, (x + 280, y + 5))
    
    def draw_statistics(self):
        """Draw overall statistics"""
        y = self.height - 120
        
        pygame.draw.line(self.screen, self.colors['grid'], (20, y), (self.width-20, y), 1)
        
        stats_header = self.font_header.render("📊 SESSION STATISTICS", True, self.colors['accent'])
        self.screen.blit(stats_header, (30, y + 10))
        
        y += 40
        
        # Calculate averages
        avg_price = np.mean(self.all_prices_history) if self.all_prices_history else 0
        
        stats = [
            f"💰 Total Revenue: ${self.total_revenue:,.2f}",
            f"🚗 Total Cars: {self.total_cars_served}",
            f"📈 Avg Price: ${avg_price:.2f}",
            f"⏱️ Session Duration: {(self.current_time - self.session_start).total_seconds()/3600:.1f} hours",
        ]
        
        for i, stat in enumerate(stats):
            t = self.font_normal.render(stat, True, self.colors['text'])
            self.screen.blit(t, (30 + i*350, y))
        
        # Controls
        controls = "Controls: SPACE=Pause/Resume | ESC=Exit"
        ctrl_t = self.font_tiny.render(controls, True, self.colors['text_dim'])
        self.screen.blit(ctrl_t, (30, self.height - 20))
    
    def draw_car_tooltip(self):
        """Draw tooltip for hovered car"""
        car = self.hovered_car
        x, y = self.mouse_pos
        
        texts = [
            f"Car #{car.car_id}",
            f"Price: ${car.assigned_price:.2f}",
            f"Duration: {car.duration_hours:.1f}h",
            f"Parked: {(self.current_time - car.entry_time).total_seconds()/60:.0f}m ago",
        ]
        
        # Tooltip box
        box_width = 180
        box_height = len(texts) * 22 + 10
        
        box_x = min(x + 15, self.width - box_width - 10)
        box_y = min(y + 15, self.height - box_height - 10)
        
        pygame.draw.rect(self.screen, (40, 40, 60), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, (100, 200, 100), (box_x, box_y, box_width, box_height), 2)
        
        for i, text in enumerate(texts):
            t = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(t, (box_x + 8, box_y + 5 + i*22))
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("🅿️  PARKING LOT DYNAMIC PRICING DEMO - FACULTY PRESENTATION")
        print("="*80)
        print("\nDemonstrating:")
        print("  ✓ Real-time parking lot simulation")
        print("  ✓ Dynamic pricing algorithm with multiple factors")
        print("  ✓ Weather and time-based effects")
        print("  ✓ Occupancy-based quadratic pricing")
        print("  ✓ Peak hour surcharges")
        print("  ✓ Weekend premium pricing")
        print("\nControls:")
        print("  • SPACE: Pause/Resume simulation")
        print("  • Hover over cars: See details")
        print("  • ESC: Exit demo")
        print("\n" + "="*80 + "\n")
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        print("\n" + "="*80)
        print("📊 DEMO SUMMARY")
        print("="*80)
        print(f"Total Revenue Generated: ${self.total_revenue:,.2f}")
        print(f"Total Cars Served: {self.total_cars_served}")
        print(f"Average Price per Car: ${np.mean(self.all_prices_history) if self.all_prices_history else 0:.2f}")
        print(f"Min Price: ${min(self.all_prices_history) if self.all_prices_history else 0:.2f}")
        print(f"Max Price: ${max(self.all_prices_history) if self.all_prices_history else 0:.2f}")
        print("="*80 + "\n")
        
        pygame.quit()


if __name__ == "__main__":
    simulator = DemoSimulator()
    simulator.run()
