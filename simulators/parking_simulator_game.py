 

import pygame
import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from role_1.env import ParkingPricingEnv
    from role_2.a2c_new import AdvancedActorCriticAgent
except ImportError:
    print("Warning: Could not import RL components. Using random pricing.")


@dataclass
class Car:
    """Represents a parked car"""
    car_id: int
    entry_time: datetime
    duration_hours: float  # How long it will stay
    assigned_price: float  # Price assigned when entered
    x: int  # Grid position x
    y: int  # Grid position y
    status: str = "parked"  # parked, leaving
    revenue_generated: float = 0.0
    
    def get_age_minutes(self, current_time: datetime) -> float:
        """Get how long the car has been parked"""
        return (current_time - self.entry_time).total_seconds() / 60
    
    def is_leaving(self, current_time: datetime) -> bool:
        """Check if car should leave"""
        age_minutes = self.get_age_minutes(current_time)
        return age_minutes >= (self.duration_hours * 60)


class ParkingLot:
    """Manages the parking lot state"""
    
    def __init__(self, rows: int = 5, cols: int = 8):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.cars: Dict[int, Car] = {}
        self.next_car_id = 1000
        self.occupancy_history: List[float] = []
        
    def get_occupancy(self) -> float:
        """Get current occupancy rate (0-1)"""
        total_spots = self.rows * self.cols
        occupied = len(self.cars)
        return occupied / total_spots if total_spots > 0 else 0
    
    def get_occupancy_percentage(self) -> float:
        """Get occupancy as percentage"""
        return self.get_occupancy() * 100
    
    def get_available_spots(self) -> List[Tuple[int, int]]:
        """Get list of available parking spots"""
        available = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] is None:
                    available.append((r, c))
        return available
    
    def can_add_car(self) -> bool:
        """Check if there are available spots"""
        return len(self.get_available_spots()) > 0
    
    def add_car(self, car: Car) -> bool:
        """Add car to parking lot"""
        if not self.can_add_car():
            return False
        
        available = self.get_available_spots()
        spot = random.choice(available)
        car.x, car.y = spot
        self.grid[spot[0]][spot[1]] = car.car_id
        self.cars[car.car_id] = car
        return True
    
    def remove_car(self, car_id: int) -> bool:
        """Remove car from parking lot"""
        if car_id not in self.cars:
            return False
        
        car = self.cars[car_id]
        self.grid[car.x][car.y] = None
        del self.cars[car_id]
        return True
    
    def get_car_at(self, x: int, y: int) -> Optional[Car]:
        """Get car at specific grid position"""
        if 0 <= x < self.rows and 0 <= y < self.cols:
            car_id = self.grid[x][y]
            if car_id and car_id in self.cars:
                return self.cars[car_id]
        return None
    
    def get_state_for_agent(self) -> np.ndarray:
        """Get state vector for RL agent"""
        occupancy = self.get_occupancy()
        num_cars = len(self.cars)
        avg_duration = (np.mean([c.duration_hours for c in self.cars.values()]) 
                       if self.cars else 0)
        
        return np.array([occupancy, num_cars / (self.rows * self.cols), avg_duration])


class RevenueTracker:
    """Tracks daily and historical revenue"""
    
    def __init__(self, data_file: str = "revenue_history.json"):
        self.data_file = Path(data_file)
        self.current_day_revenue = 0.0
        self.current_day_cars = 0
        self.history: Dict[str, Dict] = self._load_history()
        
    def _load_history(self) -> Dict:
        """Load revenue history from file"""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_day(self, date: str = None) -> None:
        """Save current day's revenue"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.history[date] = {
            "total_revenue": self.current_day_revenue,
            "total_cars": self.current_day_cars,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_revenue(self, amount: float) -> None:
        """Add revenue for current day"""
        self.current_day_revenue += amount
        self.current_day_cars += 1
    
    def get_previous_day_revenue(self, date: str = None) -> Tuple[float, str]:
        """Get previous day's revenue"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        current = datetime.strptime(date, "%Y-%m-%d")
        prev_date = (current - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if prev_date in self.history:
            return self.history[prev_date]["total_revenue"], prev_date
        return 0.0, prev_date
    
    def get_revenue_comparison(self, date: str = None) -> Dict:
        """Get comparison with previous day"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        prev_revenue, prev_date = self.get_previous_day_revenue(date)
        current_revenue = self.current_day_revenue
        
        difference = current_revenue - prev_revenue
        percentage = (difference / prev_revenue * 100) if prev_revenue > 0 else 0
        
        return {
            "current_day": date,
            "current_revenue": current_revenue,
            "previous_day": prev_date,
            "previous_revenue": prev_revenue,
            "difference": difference,
            "percentage_change": percentage
        }


class PricingAgent:
    """Determines dynamic prices for incoming cars"""
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 20.0
        self.min_price = 2.0
        
    def calculate_price(self, occupancy: float, num_cars: int, 
                       max_capacity: int, hour_of_day: int) -> float:
        """
        Calculate dynamic price based on:
        - Current occupancy
        - Time of day (peak hours)
        - Number of cars
        """
        # Base calculation from occupancy
        occupancy_factor = occupancy ** 2  # Quadratic for more sensitivity
        
        # Peak hours multiplier (9-12, 12-14, 17-20)
        peak_multiplier = 1.0
        if hour_of_day in [9, 10, 11, 12, 13, 17, 18, 19]:
            peak_multiplier = 1.3
        
        # Calculate price
        price = self.base_price + (occupancy_factor * (self.max_price - self.base_price))
        price *= peak_multiplier
        
        # Clamp to min/max
        price = max(self.min_price, min(self.max_price, price))
        
        return round(price, 2)


class ParkingSimulatorGame:
    """Main game class"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dynamic Parking Lot Simulator")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # Game state
        self.parking_lot = ParkingLot(rows=5, cols=8)
        self.pricing_agent = PricingAgent()
        self.revenue_tracker = RevenueTracker()
        self.running = True
        self.paused = False
        self.simulation_speed = 1.0  # Multiplier for time
        self.simulation_time = datetime.now()
        
        # UI state
        self.hovered_car: Optional[Car] = None
        self.mouse_pos = (0, 0)
        self.message = ""
        self.message_time = 0
        
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
        
        # Parking lot display area
        self.lot_start_x = 50
        self.lot_start_y = 100
        self.cell_size = 60
        self.lot_width = self.parking_lot.cols * self.cell_size
        self.lot_height = self.parking_lot.rows * self.cell_size
        
        # Gate button area
        self.gate_button = pygame.Rect(self.width - 250, 450, 200, 50)
        
        print("🎮 Parking Simulator Game initialized!")
        print("Controls: Click 'ADD CAR' to spawn cars, hover/click cars for details")
    
    def show_message(self, text: str, duration: float = 2.0) -> None:
        """Show temporary message"""
        self.message = text
        self.message_time = duration
    
    def update_simulation_time(self, dt: float) -> None:
        """Update simulation time"""
        self.simulation_time += timedelta(seconds=dt * self.simulation_speed)
    
    def add_random_car(self) -> bool:
        """Add a random car to parking lot"""
        if not self.parking_lot.can_add_car():
            self.show_message("⚠️  Parking lot is FULL!", 2.0)
            return False
        
        # Calculate price based on current state
        occupancy = self.parking_lot.get_occupancy()
        num_cars = len(self.parking_lot.cars)
        max_capacity = self.parking_lot.rows * self.parking_lot.cols
        hour = self.simulation_time.hour
        
        price = self.pricing_agent.calculate_price(occupancy, num_cars, max_capacity, hour)
        
        # Create car with random duration (30 mins to 3 hours)
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
            self.revenue_tracker.add_revenue(price)
            self.show_message(f"✅ Car #{car.car_id} entered at ${price:.2f}", 1.5)
            return True
        
        return False
    
    def update(self, dt: float) -> None:
        """Update game state"""
        if self.paused:
            return
        
        self.update_simulation_time(dt)
        
        # Update message timer
        if self.message_time > 0:
            self.message_time -= dt
        
        # Check for cars that should leave
        cars_to_remove = []
        for car_id, car in self.parking_lot.cars.items():
            if car.is_leaving(self.simulation_time):
                cars_to_remove.append(car_id)
                car.revenue_generated = car.assigned_price
        
        for car_id in cars_to_remove:
            self.parking_lot.remove_car(car_id)
    
    def handle_events(self) -> None:
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.hovered_car = self.get_car_at_mouse()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicked on gate button
                if self.gate_button.collidepoint(event.pos):
                    self.add_random_car()
                
                # Check if clicked on car
                elif self.hovered_car:
                    self.show_message(
                        f"Car #{self.hovered_car.car_id}: ${self.hovered_car.assigned_price:.2f}",
                        3.0
                    )
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(3.0, self.simulation_speed + 0.5)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.5)
                elif event.key == pygame.K_s:
                    self.revenue_tracker.save_day()
                    self.show_message("💾 Revenue saved!", 2.0)
                elif event.key == pygame.K_r:
                    self.reset_day()
    
    def reset_day(self) -> None:
        """Reset for new day"""
        self.revenue_tracker.save_day()
        self.parking_lot = ParkingLot(rows=5, cols=8)
        self.revenue_tracker = RevenueTracker()
        self.simulation_time = datetime.now()
        self.show_message("🆕 New day started!", 2.0)
    
    def get_car_at_mouse(self) -> Optional[Car]:
        """Get car under mouse cursor"""
        x, y = self.mouse_pos
        
        # Check if mouse is in parking lot area
        if not (self.lot_start_x <= x <= self.lot_start_x + self.lot_width and
                self.lot_start_y <= y <= self.lot_start_y + self.lot_height):
            return None
        
        # Convert to grid coordinates
        grid_x = (x - self.lot_start_x) // self.cell_size
        grid_y = (y - self.lot_start_y) // self.cell_size
        
        return self.parking_lot.get_car_at(int(grid_x), int(grid_y))
    
    def draw(self) -> None:
        """Draw game screen"""
        self.screen.fill(self.colors["bg"])
        
        # Draw title
        title = self.font_large.render("🚗 Dynamic Parking Lot Simulator 🅿️", True, self.colors["text"])
        self.screen.blit(title, (20, 20))
        
        # Draw parking lot
        self.draw_parking_lot()
        
        # Draw gate button
        self.draw_gate_button()
        
        # Draw sidebar info
        self.draw_info_panel()
        
        # Draw hover tooltip
        if self.hovered_car:
            self.draw_car_tooltip()
        
        # Draw message
        if self.message_time > 0:
            self.draw_message()
        
        # Draw controls
        self.draw_controls()
        
        pygame.display.flip()
    
    def draw_parking_lot(self) -> None:
        """Draw the parking lot grid and cars"""
        # Draw grid background
        for r in range(self.parking_lot.rows):
            for c in range(self.parking_lot.cols):
                x = self.lot_start_x + c * self.cell_size
                y = self.lot_start_y + r * self.cell_size
                
                pygame.draw.rect(self.screen, self.colors["empty"],
                               (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors["grid"],
                               (x, y, self.cell_size, self.cell_size), 2)
        
        # Draw cars
        for car in self.parking_lot.cars.values():
            x = self.lot_start_x + car.y * self.cell_size + 5
            y = self.lot_start_y + car.x * self.cell_size + 5
            size = self.cell_size - 10
            
            # Highlight if hovered
            color = (self.colors["car_hover"] if car == self.hovered_car 
                    else self.colors["car"])
            
            pygame.draw.rect(self.screen, color, (x, y, size, size))
            pygame.draw.rect(self.screen, self.colors["text"], (x, y, size, size), 2)
            
            # Draw car ID
            car_text = self.font_small.render(str(car.car_id), True, (255, 255, 255))
            text_rect = car_text.get_rect(center=(x + size//2, y + size//2))
            self.screen.blit(car_text, text_rect)
    
    def draw_gate_button(self) -> None:
        """Draw the 'ADD CAR' button"""
        color = self.colors["orange"] if self.gate_button.collidepoint(self.mouse_pos) else self.colors["blue"]
        pygame.draw.rect(self.screen, color, self.gate_button)
        pygame.draw.rect(self.screen, self.colors["text"], self.gate_button, 2)
        
        text = self.font_medium.render("➕ ADD CAR", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.gate_button.center)
        self.screen.blit(text, text_rect)
    
    def draw_info_panel(self) -> None:
        """Draw info panel on the right side"""
        x = self.lot_start_x + self.lot_width + 40
        y = 120
        
        # Occupancy
        occupancy = self.parking_lot.get_occupancy_percentage()
        capacity = self.parking_lot.rows * self.parking_lot.cols
        num_cars = len(self.parking_lot.cars)
        
        occupancy_text = f"📊 Occupancy: {num_cars}/{capacity} ({occupancy:.1f}%)"
        text = self.font_medium.render(occupancy_text, True, self.colors["text"])
        self.screen.blit(text, (x, y))
        
        # Revenue info
        y += 50
        revenue_text = f"💰 Today's Revenue: ${self.revenue_tracker.current_day_revenue:.2f}"
        text = self.font_medium.render(revenue_text, True, self.colors["green"])
        self.screen.blit(text, (x, y))
        
        # Compare with previous day
        y += 40
        comparison = self.revenue_tracker.get_revenue_comparison()
        prev_revenue = comparison["previous_revenue"]
        difference = comparison["difference"]
        percentage = comparison["percentage_change"]
        
        if prev_revenue > 0:
            color = self.colors["green"] if difference >= 0 else self.colors["red"]
            symbol = "📈" if difference >= 0 else "📉"
            prev_text = f"{symbol} vs Yesterday: ${prev_revenue:.2f}"
            text = self.font_small.render(prev_text, True, self.colors["text"])
            self.screen.blit(text, (x, y))
            
            y += 25
            diff_text = f"   Change: ${difference:+.2f} ({percentage:+.1f}%)"
            text = self.font_small.render(diff_text, True, color)
            self.screen.blit(text, (x, y))
        
        # Time and status
        y += 50
        time_text = self.simulation_time.strftime("%H:%M:%S")
        text = self.font_medium.render(f"⏰ {time_text}", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        
        y += 35
        speed_text = f"⚡ Speed: {self.simulation_speed:.1f}x"
        status = "(PAUSED)" if self.paused else "(RUNNING)"
        text = self.font_small.render(f"{speed_text} {status}", True, self.colors["text"])
        self.screen.blit(text, (x, y))
    
    def draw_car_tooltip(self) -> None:
        """Draw tooltip for hovered car"""
        car = self.hovered_car
        x, y = self.mouse_pos
        
        # Prepare tooltip text
        duration = car.duration_hours
        age = car.get_age_minutes(self.simulation_time)
        remaining = max(0, duration * 60 - age)
        
        lines = [
            f"Car #{car.car_id}",
            f"Price: ${car.assigned_price:.2f}",
            f"Duration: {duration:.1f}h",
            f"Parked: {age:.0f}min",
            f"Remaining: {remaining:.0f}min"
        ]
        
        # Draw tooltip background
        max_width = max(len(line) for line in lines) * 8
        tooltip_height = len(lines) * 22 + 10
        
        tooltip_x = min(x + 10, self.width - max_width - 10)
        tooltip_y = min(y + 10, self.height - tooltip_height - 10)
        
        pygame.draw.rect(self.screen, (50, 50, 50),
                        (tooltip_x, tooltip_y, max_width, tooltip_height))
        pygame.draw.rect(self.screen, self.colors["text"],
                        (tooltip_x, tooltip_y, max_width, tooltip_height), 1)
        
        # Draw text
        for i, line in enumerate(lines):
            text = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(text, (tooltip_x + 5, tooltip_y + 5 + i * 22))
    
    def draw_message(self) -> None:
        """Draw temporary message"""
        text = self.font_medium.render(self.message, True, self.colors["green"])
        text_rect = text.get_rect(center=(self.width // 2, 60))
        
        # Draw background
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect)
        pygame.draw.rect(self.screen, self.colors["green"], bg_rect, 2)
        
        self.screen.blit(text, text_rect)
    
    def draw_controls(self) -> None:
        """Draw control instructions"""
        instructions = [
            "CONTROLS:",
            "CLICK → Add car at gate",
            "HOVER → Car details",
            "SPACE → Pause/Resume",
            "↑/↓ → Speed up/down",
            "S → Save revenue",
            "R → New day"
        ]
        
        x = self.width - 250
        y = self.height - 200
        
        for i, instruction in enumerate(instructions):
            color = self.colors["blue"] if i == 0 else self.colors["text"]
            text = self.font_small.render(instruction, True, color)
            self.screen.blit(text, (x, y + i * 25))
    
    def run(self) -> None:
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Convert to seconds
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        # Save on exit
        self.revenue_tracker.save_day()
        pygame.quit()
        print("✅ Game saved and closed!")


def main():
    """Main entry point"""
    game = ParkingSimulatorGame()
    game.run()


if __name__ == "__main__":
    main()
