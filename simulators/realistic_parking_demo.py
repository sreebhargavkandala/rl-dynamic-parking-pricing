 

import pygame
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

pygame.init()

 
WIDTH, HEIGHT = 1800, 1000
FPS = 60

COLORS = {
    'bg': (240, 240, 245),
    'asphalt': (60, 60, 70),
    'parking_line': (255, 255, 100),
    'grass': (100, 200, 100),
    'car': (255, 100, 100),
    'car_hover': (255, 200, 100),
    'gate': (100, 100, 255),
    'text': (30, 30, 30),
    'text_light': (100, 100, 100),
    'price_good': (76, 175, 80),
    'price_high': (255, 152, 0),
    'price_peak': (244, 67, 54),
}

 
class Car:
    """Realistic car representation"""
    def __init__(self, car_id, x, y, price, duration, entry_time):
        self.car_id = car_id
        self.x = x
        self.y = y
        self.price = price
        self.duration = duration
        self.entry_time = entry_time
        self.width = 40
        self.height = 25
        self.angle = 0
        self.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    
    def draw(self, surface, hovered=False):
        """Draw car with 3D-like perspective"""
        # Car body
        if hovered:
            pygame.draw.rect(surface, (255, 255, 0), (self.x-2, self.y-2, self.width+4, self.height+4), 3)
        
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))
        
        # Windows
        window_color = (100, 150, 200)
        pygame.draw.rect(surface, window_color, (self.x+5, self.y+3, 12, 8))
        pygame.draw.rect(surface, window_color, (self.x+23, self.y+3, 12, 8))
        
        # Wheels
        wheel_color = (20, 20, 20)
        pygame.draw.circle(surface, wheel_color, (int(self.x+8), int(self.y+self.height)), 3)
        pygame.draw.circle(surface, wheel_color, (int(self.x+32), int(self.y+self.height)), 3)
        
        # License plate
        pygame.draw.rect(surface, (200, 200, 200), (self.x+15, self.y+self.height-5, 10, 3))
        
        # Car ID
        font = pygame.font.Font(None, 14)
        id_text = font.render(str(self.car_id % 100), True, (255, 255, 255))
        surface.blit(id_text, (self.x+10, self.y+8))
    
    def draw_info_box(self, surface, mouse_pos):
        """Draw info box when hovered"""
        if not (self.x <= mouse_pos[0] <= self.x + self.width and
                self.y <= mouse_pos[1] <= self.y + self.height):
            return False
        
        font_small = pygame.font.Font(None, 16)
        
        lines = [
            f"Car #{self.car_id}",
            f"Price: ${self.price:.2f}",
            f"Duration: {self.duration:.1f}h"
        ]
        
        box_x = mouse_pos[0] + 10
        box_y = mouse_pos[1] - 50
        
        pygame.draw.rect(surface, (40, 40, 40), (box_x, box_y, 140, 70))
        pygame.draw.rect(surface, (255, 255, 100), (box_x, box_y, 140, 70), 2)
        
        for i, line in enumerate(lines):
            text = font_small.render(line, True, (255, 255, 255))
            surface.blit(text, (box_x + 8, box_y + 8 + i*20))
        
        return True


 
class ParkingSpace:
    """Individual parking space"""
    def __init__(self, x, y, space_id):
        self.x = x
        self.y = y
        self.space_id = space_id
        self.occupied = False
        self.car = None
        self.width = 50
        self.height = 35
    
    def draw(self, surface):
        """Draw parking space with markings"""
        # Space background
        pygame.draw.rect(surface, (200, 200, 200), (self.x, self.y, self.width, self.height))
        
        # White border lines (parking space markings)
        pygame.draw.rect(surface, COLORS['parking_line'], (self.x, self.y, self.width, self.height), 2)
        
        # Space number
        if not self.occupied:
            font = pygame.font.Font(None, 12)
            num_text = font.render(str(self.space_id), True, COLORS['text_light'])
            surface.blit(num_text, (self.x + 8, self.y + 10))
    
    def is_inside(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


class ParkingLot:
    """Complete parking lot with spaces"""
    def __init__(self):
        self.spaces = []
        self.cars = {}
        self.car_counter = 1000
        
        # Create 4 rows x 10 columns = 40 spaces
        start_x = 300
        start_y = 200
        space_spacing_x = 60
        space_spacing_y = 50
        
        space_id = 1
        for row in range(4):
            for col in range(10):
                x = start_x + col * space_spacing_x
                y = start_y + row * space_spacing_y
                self.spaces.append(ParkingSpace(x, y, space_id))
                space_id += 1
    
    def get_occupancy(self):
        occupied = sum(1 for space in self.spaces if space.occupied)
        return occupied / len(self.spaces)
    
    def get_free_space(self):
        for space in self.spaces:
            if not space.occupied:
                return space
        return None
    
    def add_car(self, car):
        space = self.get_free_space()
        if space:
            space.occupied = True
            space.car = car
            car.x = space.x + 5
            car.y = space.y + 5
            self.cars[car.car_id] = car
            return True
        return False
    
    def remove_car(self, car_id):
        if car_id in self.cars:
            car = self.cars[car_id]
            # Find and free the space
            for space in self.spaces:
                if space.car and space.car.car_id == car_id:
                    space.occupied = False
                    space.car = None
                    break
            del self.cars[car_id]
            return True
        return False
    
    def draw(self, surface):
        for space in self.spaces:
            space.draw(surface)
        
        for car in self.cars.values():
            car.draw(surface)
    
    def get_car_at_mouse(self, pos):
        for car in self.cars.values():
            if car.x <= pos[0] <= car.x + car.width and car.y <= pos[1] <= car.y + car.height:
                return car
        return None


 

class DynamicPricingEngine:
    """Dynamic pricing with multiple factors"""
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 25.0
        self.min_price = 1.5
        
        self.weather_effects = {
            "Sunny": 1.0, "Cloudy": 0.95, "Rainy": 0.85,
            "Snowy": 0.70, "Foggy": 0.90
        }
        
        self.peak_hours = [8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
        self.semi_peak = [7, 11, 15, 16, 21]
    
    def calculate_price(self, occupancy, hour, is_weekend, weather):
        """Calculate final price with breakdown"""
        
        # Occupancy factor (quadratic - MAIN DRIVER)
        occupancy_factor = occupancy ** 2
        occupancy_boost = occupancy_factor * (self.max_price - self.base_price)
        
        # Peak hour multiplier
        if hour in self.peak_hours:
            peak_mult = 1.4
        elif hour in self.semi_peak:
            peak_mult = 1.2
        else:
            peak_mult = 0.85
        
        # Weekend multiplier
        weekend_mult = 1.15 if is_weekend else 1.0
        
        # Weather effect
        weather_effect = self.weather_effects.get(weather, 1.0)
        
        # Calculate final price
        price = self.base_price + occupancy_boost
        price = price * peak_mult * weekend_mult * weather_effect
        final_price = max(self.min_price, min(self.max_price, round(price, 2)))
        
        return final_price, {
            'occupancy': occupancy,
            'occupancy_boost': occupancy_boost,
            'peak_mult': peak_mult,
            'weekend_mult': weekend_mult,
            'weather_effect': weather_effect,
            'base': self.base_price
        }


 

class RealisticParkingSimulator:
    """Main simulator with realistic graphics"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🅿️ REALISTIC PARKING LOT SIMULATOR - Dynamic Pricing Demo")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Components
        self.parking_lot = ParkingLot()
        self.pricing_engine = DynamicPricingEngine()
        
        # Time management
        self.current_time = datetime.now().replace(hour=8, minute=0, second=0)
        self.session_start = self.current_time
        self.last_weather_update = self.current_time.hour
        
        # Weather
        self.weather_list = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
        self.current_weather = "Sunny"
        self.weather_emoji = {
            "Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️",
            "Snowy": "❄️", "Foggy": "🌫️"
        }
        
        # Tracking
        self.total_revenue = 0.0
        self.total_cars = 0
        self.prices_history = []
        self.last_add_time = 0
        self.add_interval = 2.0  # Slower - adds cars every 2 seconds
        self.simulation_speed = 0.3  # SLOW! 0.3 hours per second for clear visibility
        
        # UI
        self.hovered_car = None
        self.mouse_pos = (0, 0)
        self.message = ""
        self.message_time = 0
        
        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_header = pygame.font.Font(None, 28)
        self.font_normal = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
    
    def update_weather(self):
        """Update weather randomly"""
        if self.current_time.hour != self.last_weather_update:
            self.last_weather_update = self.current_time.hour
            self.current_weather = random.choice(self.weather_list)
    
    def add_cars(self):
        """Add cars based on time and conditions"""
        hour = self.current_time.hour
        occupancy = self.parking_lot.get_occupancy()
        is_weekend = self.current_time.weekday() >= 5
        
        # Determine arrivals based on time - REDUCED FOR SLOW VIEWING
        if hour in [8, 9, 10, 17, 18, 19]:  # Peak arrival times
            num_arrivals = random.randint(1, 2)  # REDUCED from 4-6
        elif hour in [11, 12, 13, 14, 15, 20, 21]:  # Medium
            num_arrivals = random.randint(0, 2)  # REDUCED from 2-4
        else:  # Off-peak
            num_arrivals = random.randint(0, 1)  # REDUCED from 0-2
        
        # Weather effect on arrivals
        if self.current_weather == "Snowy":
            num_arrivals = int(num_arrivals * 0.6)
        elif self.current_weather == "Rainy":
            num_arrivals = int(num_arrivals * 0.75)
        elif self.current_weather == "Sunny":
            num_arrivals = int(num_arrivals * 1.15)
        
        # Don't exceed capacity
        free_spaces = len([s for s in self.parking_lot.spaces if not s.occupied])
        num_arrivals = min(num_arrivals, free_spaces)
        
        # Add cars
        for _ in range(num_arrivals):
            occupancy = self.parking_lot.get_occupancy()
            price, breakdown = self.pricing_engine.calculate_price(
                occupancy, hour, is_weekend, self.current_weather
            )
            
            car = Car(
                car_id=self.parking_lot.car_counter,
                x=0, y=0,
                price=price,
                duration=random.uniform(0.5, 3.0),
                entry_time=self.current_time
            )
            self.parking_lot.car_counter += 1
            
            if self.parking_lot.add_car(car):
                self.total_revenue += price
                self.total_cars += 1
                self.prices_history.append(price)
    
    def remove_expired_cars(self):
        """Remove cars that have finished parking"""
        cars_to_remove = []
        for car_id, car in self.parking_lot.cars.items():
            age_hours = (self.current_time - car.entry_time).total_seconds() / 3600
            if age_hours >= car.duration:
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            self.parking_lot.remove_car(car_id)
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(100, self.simulation_speed + 5)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(1, self.simulation_speed - 5)
    
    def update(self, dt):
        """Update simulation"""
        if self.paused or not self.running:
            return
        
        # Add cars periodically
        self.last_add_time += dt
        if self.last_add_time >= self.add_interval:
            self.add_cars()
            self.last_add_time = 0
        
        # Advance time
        self.current_time += timedelta(minutes=self.simulation_speed * dt * 60)
        
        # Update weather
        self.update_weather()
        
        # Remove expired cars
        self.remove_expired_cars()
        
        # Update hover
        self.hovered_car = self.parking_lot.get_car_at_mouse(self.mouse_pos)
        
        # Message fade
        if self.message_time > 0:
            self.message_time -= dt
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(COLORS['bg'])
        
        # Title bar
        self.draw_title_bar()
        
        # Main parking lot area
        self.draw_lot_background()
        
        # Parking lot and cars
        self.parking_lot.draw(self.screen)
        
        # Info panels
        self.draw_left_panel()
        self.draw_right_panel()
        
        # Hover tooltip
        if self.hovered_car:
            self.hovered_car.draw_info_box(self.screen, self.mouse_pos)
        
        # Controls and stats
        self.draw_bottom_bar()
        
        # Pause indicator
        if self.paused:
            pause_text = self.font_header.render("⏸ PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_text, (WIDTH - 300, 20))
        
        pygame.display.flip()
    
    def draw_title_bar(self):
        """Draw title bar"""
        title = self.font_title.render("🅿️ REALISTIC PARKING LOT SIMULATOR", True, COLORS['text'])
        self.screen.blit(title, (30, 15))
    
    def draw_lot_background(self):
        """Draw parking lot background"""
        # Asphalt
        pygame.draw.rect(self.screen, COLORS['asphalt'], (280, 180, 800, 400))
        
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), (280, 180, 800, 400), 3)
        
        # Entrance
        pygame.draw.rect(self.screen, COLORS['gate'], (270, 300, 15, 50))
        entrance_text = self.font_small.render("IN", True, (255, 255, 255))
        self.screen.blit(entrance_text, (275, 310))
        
        # Exit
        pygame.draw.rect(self.screen, COLORS['gate'], (1095, 300, 15, 50))
        exit_text = self.font_small.render("OUT", True, (255, 255, 255))
        self.screen.blit(exit_text, (1095, 310))
    
    def draw_left_panel(self):
        """Draw left information panel"""
        x, y = 30, 180
        
        # Current conditions
        title = self.font_header.render("CURRENT CONDITIONS", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        time_str = self.current_time.strftime("%H:%M")
        day_str = self.current_time.strftime("%A")
        
        conditions = [
            f"⏰ Time: {time_str}",
            f"📅 Day: {day_str}",
            f"🌦️ Weather: {self.weather_emoji[self.current_weather]} {self.current_weather}",
            f"",
            f"📊 Occupancy:",
            f"   {len(self.parking_lot.cars)}/40 spaces ({self.parking_lot.get_occupancy()*100:.1f}%)",
        ]
        
        for condition in conditions:
            if condition == "":
                y += 10
            else:
                text = self.font_small.render(condition, True, COLORS['text'])
                self.screen.blit(text, (x, y))
            y += 25
    
    def draw_right_panel(self):
        """Draw right statistics panel"""
        x, y = WIDTH - 320, 180
        
        # Revenue stats
        title = self.font_header.render("💰 REVENUE & STATS", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        avg_price = np.mean(self.prices_history) if self.prices_history else 0
        
        stats = [
            f"Total Revenue: ${self.total_revenue:,.2f}",
            f"Cars Served: {self.total_cars}",
            f"Avg Price: ${avg_price:.2f}",
            f"Session: {(self.current_time - self.session_start).total_seconds()/3600:.1f}h",
        ]
        
        for stat in stats:
            text = self.font_small.render(stat, True, COLORS['text'])
            self.screen.blit(text, (x, y))
            y += 30
        
        # Current pricing formula
        y += 15
        hour = self.current_time.hour
        is_weekend = self.current_time.weekday() >= 5
        occupancy = self.parking_lot.get_occupancy()
        price, breakdown = self.pricing_engine.calculate_price(
            occupancy, hour, is_weekend, self.current_weather
        )
        
        formula_title = self.font_header.render("Current Price Calculation:", True, COLORS['text'])
        self.screen.blit(formula_title, (x, y))
        y += 35
        
        formula_lines = [
            f"Base: ${breakdown['base']:.2f}",
            f"+ Occupancy²: ${breakdown['occupancy_boost']:.2f}",
            f"× Peak (h={hour}): {breakdown['peak_mult']:.2f}x",
            f"× Weather: {breakdown['weather_effect']:.2f}x",
            f"× Weekend: {breakdown['weekend_mult']:.2f}x",
        ]
        
        for line in formula_lines:
            text = self.font_small.render(line, True, COLORS['text_light'])
            self.screen.blit(text, (x, y))
            y += 22
        
        # Final price
        price_color = COLORS['price_peak'] if price > 20 else COLORS['price_high'] if price > 15 else COLORS['price_good']
        final_text = self.font_header.render(f"= ${price:.2f}", True, price_color)
        self.screen.blit(final_text, (x, y))
    
    def draw_bottom_bar(self):
        """Draw bottom information bar"""
        y = HEIGHT - 70
        
        # Background
        pygame.draw.rect(self.screen, (200, 200, 200), (0, y, WIDTH, 70))
        pygame.draw.line(self.screen, (100, 100, 100), (0, y), (WIDTH, y), 2)
        
        # Controls
        controls_text = "CONTROLS: SPACE=Pause | ↑↓=Speed | ESC=Exit"
        controls = self.font_small.render(controls_text, True, COLORS['text'])
        self.screen.blit(controls, (30, y + 15))
        
        # Simulation speed
        speed_text = f"Speed: {self.simulation_speed}x"
        speed = self.font_small.render(speed_text, True, COLORS['text'])
        self.screen.blit(speed, (WIDTH - 250, y + 15))
        
        # Instructions
        instr_text = "💡 Hover over cars to see details | Watch prices change with occupancy, time, and weather!"
        instr = self.font_small.render(instr_text, True, COLORS['text_light'])
        self.screen.blit(instr, (30, y + 40))
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("🅿️ REALISTIC PARKING LOT SIMULATOR - FACULTY DEMO")
        print("="*80)
        print("\nShowing:")
        print("  ✓ Real parking lot layout with 40 spaces")
        print("  ✓ Realistic car arrivals and departures")
        print("  ✓ Dynamic pricing based on multiple factors")
        print("  ✓ Real-time occupancy-based pricing")
        print("  ✓ Weather effects on demand")
        print("  ✓ Peak hour surcharges")
        print("\nWatch how prices change as occupancy changes!")
        print("="*80 + "\n")
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        print(f"Total Revenue: ${self.total_revenue:,.2f}")
        print(f"Total Cars Parked: {self.total_cars}")
        if self.prices_history:
            print(f"Avg Price: ${np.mean(self.prices_history):.2f}")
            print(f"Min Price: ${min(self.prices_history):.2f}")
            print(f"Max Price: ${max(self.prices_history):.2f}")
        print("="*80 + "\n")
        
        pygame.quit()


if __name__ == "__main__":
    simulator = RealisticParkingSimulator()
    simulator.run()
