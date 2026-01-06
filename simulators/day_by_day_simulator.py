 

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
    'button': (70, 130, 180),
    'button_hover': (100, 150, 200),
    'button_text': (255, 255, 255),
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
            surface.blit(text, (box_x + 8, box_y + 5 + i*20))
        
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


 
class Button:
    """Clickable button"""
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
    
    def draw(self, surface, font):
        color = COLORS['button_hover'] if self.hovered else COLORS['button']
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 2)
        
        text_surface = font.render(self.text, True, COLORS['button_text'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


 
class DayByDaySimulator:
    """Main simulator with day-by-day progression"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🅿️ PARKING LOT SIMULATOR - Day by Day")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Components
        self.parking_lot = ParkingLot()
        self.pricing_engine = DynamicPricingEngine()
        
        # Day management
        self.current_day_num = 1
        self.current_date = datetime.now().replace(hour=6, minute=0, second=0)
        self.day_start_time = self.current_date
        self.day_end_time = self.current_date.replace(hour=18)  # 6 PM
        self.current_time = self.day_start_time
        
        # Day status
        self.day_complete = False
        self.day_revenue = 0.0
        self.day_cars = 0
        self.previous_day_revenue = 0.0
        self.previous_day_cars = 0
        
        # Weather
        self.weather_list = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
        self.current_weather = random.choice(self.weather_list)
        self.weather_emoji = {
            "Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️",
            "Snowy": "❄️", "Foggy": "🌫️"
        }
        
        # Tracking
        self.total_revenue = 0.0
        self.total_cars = 0
        self.prices_history = []
        self.last_add_time = 0
        self.add_interval = 0.8  # Slower arrivals
        self.simulation_speed = 3.0  # 3 hours per second (completes 12-hour day in 4 seconds)
        
        # UI
        self.hovered_car = None
        self.mouse_pos = (0, 0)
        self.next_day_button = Button(WIDTH - 220, 20, 200, 50, "► NEXT DAY")
        
        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_header = pygame.font.Font(None, 28)
        self.font_normal = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_button = pygame.font.Font(None, 24)
    
    def add_cars(self):
        """Add cars based on time and conditions"""
        hour = self.current_time.hour
        occupancy = self.parking_lot.get_occupancy()
        is_weekend = self.current_time.weekday() >= 5
        
        # INCREASED arrivals to fill lot properly
        if hour in [8, 9, 10, 12, 13, 14, 17, 18]:  # Peak
            num_arrivals = random.randint(3, 5)  # MORE cars
        elif hour in [7, 11, 15, 16]:  # Semi-peak
            num_arrivals = random.randint(2, 3)
        else:  # Off-peak
            num_arrivals = random.randint(1, 2)
        
        # Weather effect
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
                self.day_revenue += price
                self.total_revenue += price
                self.day_cars += 1
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
    
    def start_next_day(self):
        """Start the next day"""
        # Save previous day stats
        self.previous_day_revenue = self.day_revenue
        self.previous_day_cars = self.day_cars
        
        # Reset for new day
        self.current_day_num += 1
        self.current_date += timedelta(days=1)
        self.day_start_time = self.current_date.replace(hour=6)
        self.day_end_time = self.current_date.replace(hour=18)
        self.current_time = self.day_start_time
        self.day_revenue = 0.0
        self.day_cars = 0
        self.day_complete = False
        self.parking_lot = ParkingLot()
        self.current_weather = random.choice(self.weather_list)
        self.paused = False
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.next_day_button.update_hover(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.next_day_button.is_clicked(event.pos) and self.day_complete:
                    self.start_next_day()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self, dt):
        """Update simulation"""
        if self.paused or not self.running or self.day_complete:
            return
        
        # Add cars periodically
        self.last_add_time += dt
        if self.last_add_time >= self.add_interval:
            self.add_cars()
            self.last_add_time = 0
        
        # Advance time
        self.current_time += timedelta(minutes=self.simulation_speed * dt * 60)
        
        # Check if day is complete
        if self.current_time >= self.day_end_time:
            self.current_time = self.day_end_time
            self.day_complete = True
            self.paused = True
        
        # Remove expired cars
        self.remove_expired_cars()
        
        # Update hover
        self.hovered_car = self.parking_lot.get_car_at_mouse(self.mouse_pos)
    
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
        self.draw_previous_day_corner()
        
        # Hover tooltip
        if self.hovered_car:
            self.hovered_car.draw_info_box(self.screen, self.mouse_pos)
        
        # Controls and stats
        self.draw_bottom_bar()
        
        # Next day button
        self.next_day_button.draw(self.screen, self.font_button)
        
        # Day complete message
        if self.day_complete:
            self.draw_day_complete()
        
        # Pause indicator
        if self.paused and not self.day_complete:
            pause_text = self.font_header.render("⏸ PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_text, (WIDTH - 300, 85))
        
        pygame.display.flip()
    
    def draw_title_bar(self):
        """Draw title bar"""
        title = self.font_title.render(f"🅿️ PARKING LOT SIMULATOR - Day {self.current_day_num}", True, COLORS['text'])
        self.screen.blit(title, (30, 20))
    
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
        title = self.font_header.render("📅 TODAY'S CONDITIONS", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        time_str = self.current_time.strftime("%H:%M")
        day_str = self.current_date.strftime("%A, %B %d")
        
        conditions = [
            f"⏰ Time: {time_str}",
            f"📅 Date: {day_str}",
            f"🌦️ Weather: {self.weather_emoji[self.current_weather]} {self.current_weather}",
            f"",
            f"📊 Lot Status:",
            f"   {len(self.parking_lot.cars)}/40 spaces ({self.parking_lot.get_occupancy()*100:.1f}%)",
            f"",
            f"💰 Today Revenue: ${self.day_revenue:,.2f}",
            f"🚗 Today Cars: {self.day_cars}",
        ]
        
        for condition in conditions:
            if condition == "":
                y += 10
            else:
                color = (100, 200, 100) if "Revenue" in condition else COLORS['text']
                text = self.font_small.render(condition, True, color)
                self.screen.blit(text, (x, y))
            y += 25
    
    def draw_previous_day_corner(self):
        """Draw previous day stats in corner"""
        x, y = 30, HEIGHT - 180
        
        # Background box
        pygame.draw.rect(self.screen, (240, 240, 255), (x, y, 300, 160))
        pygame.draw.rect(self.screen, (100, 100, 200), (x, y, 300, 160), 3)
        
        # Title
        title = self.font_header.render("📈 PREVIOUS DAY", True, (50, 50, 150))
        self.screen.blit(title, (x + 10, y + 10))
        
        y += 45
        
        if self.current_day_num > 1:
            stats = [
                f"Revenue: ${self.previous_day_revenue:,.2f}",
                f"Cars Served: {self.previous_day_cars}",
                f"Avg Price: ${self.previous_day_revenue / self.previous_day_cars if self.previous_day_cars > 0 else 0:.2f}",
            ]
        else:
            stats = [
                f"Day 1 - No previous data",
                f"",
                f"",
            ]
        
        for stat in stats:
            if stat:
                text = self.font_normal.render(stat, True, COLORS['text'])
                self.screen.blit(text, (x + 15, y))
            y += 30
    
    def draw_right_panel(self):
        """Draw right statistics panel"""
        x, y = WIDTH - 320, 180
        
        # Revenue stats
        title = self.font_header.render("💰 CUMULATIVE STATS", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        avg_price = np.mean(self.prices_history) if self.prices_history else 0
        
        stats = [
            f"Total Revenue: ${self.total_revenue:,.2f}",
            f"Total Cars: {self.total_cars}",
            f"Avg Price: ${avg_price:.2f}",
            f"Days Run: {self.current_day_num}",
        ]
        
        for stat in stats:
            text = self.font_small.render(stat, True, COLORS['text'])
            self.screen.blit(text, (x, y))
            y += 30
        
        # Current pricing formula
        y += 15
        hour = self.current_time.hour
        is_weekend = self.current_date.weekday() >= 5
        occupancy = self.parking_lot.get_occupancy()
        price, breakdown = self.pricing_engine.calculate_price(
            occupancy, hour, is_weekend, self.current_weather
        )
        
        formula_title = self.font_header.render("Current Price:", True, COLORS['text'])
        self.screen.blit(formula_title, (x, y))
        y += 35
        
        formula_lines = [
            f"Base: ${breakdown['base']:.2f}",
            f"+ Occ²: ${breakdown['occupancy_boost']:.2f}",
            f"× Peak: {breakdown['peak_mult']:.2f}x",
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
        if self.day_complete:
            controls_text = "✅ DAY COMPLETE! Click 'NEXT DAY' to continue"
        else:
            controls_text = "CONTROLS: SPACE=Pause | ESC=Exit"
        
        controls = self.font_small.render(controls_text, True, COLORS['text'])
        self.screen.blit(controls, (30, y + 15))
        
        # Instructions
        instr_text = "💡 Hover over cars to see details | Watch how prices change throughout the day"
        instr = self.font_small.render(instr_text, True, COLORS['text_light'])
        self.screen.blit(instr, (30, y + 40))
    
    def draw_day_complete(self):
        """Draw day complete overlay"""
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Message
        complete_text = self.font_title.render("✅ DAY COMPLETE!", True, (100, 255, 100))
        complete_rect = complete_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        self.screen.blit(complete_text, complete_rect)
        
        # Stats
        stats_text = self.font_header.render(
            f"Revenue: ${self.day_revenue:,.2f} | Cars: {self.day_cars}",
            True, (255, 255, 100)
        )
        stats_rect = stats_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(stats_text, stats_rect)
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("🅿️ PARKING LOT SIMULATOR - DAY BY DAY DEMO")
        print("="*80)
        print("\nMode: Single day simulation (6 AM to 6 PM)")
        print("\nClick NEXT DAY button after each day to progress")
        print("Shows previous day revenue for comparison")
        print("="*80 + "\n")
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        print(f"Total Days Simulated: {self.current_day_num}")
        print(f"Total Revenue: ${self.total_revenue:,.2f}")
        print(f"Total Cars: {self.total_cars}")
        if self.prices_history:
            print(f"Avg Price: ${np.mean(self.prices_history):.2f}")
        print("="*80 + "\n")
        
        pygame.quit()


if __name__ == "__main__":
    simulator = DayByDaySimulator()
    simulator.run()
