 

import pygame
import random
import math
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque, deque
import sys
from pathlib import Path

# Add role_2 to path for A2C import
sys.path.insert(0, str(Path(__file__).parent.parent / 'role_2'))

try:
    from a2c_new import A2CAgent, A2CConfig
    A2C_AVAILABLE = True
except ImportError:
    A2C_AVAILABLE = False
    print("⚠️ A2C agent not available, using fallback Q-Learning")

pygame.init()

 
WIDTH, HEIGHT = 1800, 1000
FPS = 60

COLORS = {
    'bg': (240, 240, 245),
    'asphalt': (60, 60, 70),
    'parking_line': (255, 255, 100),
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
    'rl_accent': (147, 51, 234),  # Purple for RL
}

 

class A2CPricingModel:
    """
    A2C (Actor-Critic) Agent for Dynamic Pricing
    Integrates A2C agent from role_2 for advanced learning
    """
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 30.0
        self.min_price = 1.5
        
        # Initialize A2C agent
        if A2C_AVAILABLE:
            state_dim = 3  # [occupancy, hour, weather]
            action_dim = 5  # 5 price levels
            self.config = A2CConfig(state_dim=state_dim, action_dim=action_dim)
            self.agent = A2CAgent(self.config)
            self.use_a2c = True
        else:
            self.use_a2c = False
        
        # Fallback Q-Table for when A2C not available
        self.q_table = defaultdict(lambda: {})
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        
        # Training tracking
        self.days_trained = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.price_history = []
        self.reward_history = []
    
    def _discretize_state(self, occupancy, hour, weather):
        """Convert continuous state to discrete for fallback Q-Learning"""
        occ_level = min(4, int(occupancy * 5))
        hour_period = 0 if hour < 12 else (1 if hour < 18 else 2)
        return (occ_level, hour_period, weather)
    
    def _get_price_actions(self, occupancy, hour, weather):
        """Generate possible price actions"""
        time_demand = {
            6: 0.2, 7: 0.4, 8: 0.9, 9: 1.0, 10: 0.95,
            11: 0.8, 12: 0.85, 13: 0.9, 14: 0.85, 15: 0.7,
            16: 0.75, 17: 1.0, 18: 0.9, 19: 0.7, 20: 0.5, 21: 0.3
        }
        weather_demand = {
            "Sunny": 1.2, "Cloudy": 0.95, "Rainy": 0.7,
            "Snowy": 0.5, "Foggy": 0.85
        }
        
        base_boost = occupancy ** 2 * (self.max_price - self.base_price)
        hour_demand = time_demand.get(hour, 0.5)
        weather_mult = weather_demand.get(weather, 1.0)
        base_calc = self.base_price + base_boost * hour_demand * weather_mult
        
        actions = [
            max(self.min_price, base_calc * 0.7),
            max(self.min_price, base_calc * 0.85),
            max(self.min_price, base_calc * 1.0),
            max(self.min_price, base_calc * 1.2),
            min(self.max_price, base_calc * 1.4),
        ]
        return [round(p, 2) for p in actions]
    
    def get_optimal_price(self, occupancy, hour, weather, is_weekend, day_num, training=True):
        """Select optimal price using A2C or Q-Learning"""
        
        if self.use_a2c:
            # Use A2C agent
            # Normalize state to [0, 1]
            state_features = np.array([
                occupancy,  # [0, 1]
                hour / 24.0,  # [0, 1]
                {"Sunny": 0, "Cloudy": 0.25, "Rainy": 0.5, "Snowy": 0.75, "Foggy": 1.0}.get(weather, 0.5)
            ], dtype=np.float32)
            
            # Get action from A2C (continuous action)
            action = self.agent.select_action(state_features)
            
            # Map continuous action to price (action is in [0, 1])
            action_idx = int(action * 4.9) if isinstance(action, (float, np.ndarray)) else action
            action_idx = max(0, min(4, action_idx))
            
            # Get price candidates
            prices = self._get_price_actions(occupancy, hour, weather)
            price = prices[action_idx]
        else:
            # Fallback to Q-Learning
            state = self._discretize_state(occupancy, hour, weather)
            prices = self._get_price_actions(occupancy, hour, weather)
            
            if training and random.random() < self.epsilon:
                price = random.choice(prices)
            else:
                best_price = prices[2]  # Default to middle price
                best_q = -float('inf')
                for p in prices:
                    if state not in self.q_table:
                        self.q_table[state] = {}
                    q_val = self.q_table[state].get(p, 0.5)
                    if q_val > best_q:
                        best_q = q_val
                        best_price = p
                price = best_price
            
            self.epsilon = max(0.05, self.epsilon * 0.98)
        
        self.price_history.append(price)
        return price
    
    def train_on_day_reward(self, day_states, day_rewards):
        """Train A2C agent with day's rewards"""
        total_day_reward = sum(day_rewards)
        self.episode_rewards.append(total_day_reward)
        self.total_reward = total_day_reward
        
        if self.use_a2c:
            # Train A2C agent
            # Convert states to features
            state_features = []
            for state_info in day_states:
                if isinstance(state_info, tuple) and len(state_info) == 3:
                    occ, hour, weather = state_info
                else:
                    # If it's not in expected format, skip
                    continue
                
                features = np.array([
                    occ, hour / 24.0,
                    {"Sunny": 0, "Cloudy": 0.25, "Rainy": 0.5, "Snowy": 0.75, "Foggy": 1.0}.get(weather, 0.5)
                ], dtype=np.float32)
                state_features.append(features)
            
            # Train on episode
            if state_features:
                try:
                    self.agent.train_episode(
                        states=state_features,
                        actions=np.array(list(range(len(day_rewards)))) % 5,
                        rewards=np.array(day_rewards, dtype=np.float32)
                    )
                except Exception as e:
                    print(f"A2C training error: {e}")
        else:
            # Fallback Q-Learning training
            for i, state_info in enumerate(day_states[:-1]):
                if isinstance(state_info, tuple) and len(state_info) == 3:
                    state = state_info
                    reward = day_rewards[i] if i < len(day_rewards) else 0
                    next_state = day_states[i + 1] if i + 1 < len(day_states) else state
                    
                    if state not in self.q_table:
                        self.q_table[state] = {}
                    
                    # Simple Q-Learning update
                    prices = self._get_price_actions(0.5, 12, "Sunny")
                    price = prices[i % len(prices)]
                    
                    current_q = self.q_table[state].get(price, 0.5)
                    max_next_q = max(self.q_table[next_state].values()) if next_state in self.q_table else 0.5
                    new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
                    self.q_table[state][price] = new_q
        
        self.days_trained += 1
        self.learning_rate = max(0.01, self.learning_rate * 0.99)


 

class RLPricingModel:
    """
    Advanced RL-based dynamic pricing model
    Learns optimal prices using Q-Learning-inspired approach
    Improves pricing strategy over multiple days
    """
    
    def __init__(self):
        self.base_price = 5.0
        self.max_price = 30.0  # Higher for peak times
        self.min_price = 1.5
        
        # Learning parameters (actual RL learning)
        self.learning_rate = 0.1
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.price_history = []
        self.reward_history = []
        
        # Q-Table: state -> {price_action: q_value}
        # States: (occupancy_level, hour, weather)
        self.q_table = defaultdict(lambda: {})
        self.state_visits = defaultdict(int)
        self.state_rewards = defaultdict(list)
        
        # Weather and demand mappings
        self.weather_demand = {
            "Sunny": 1.2, "Cloudy": 0.95, "Rainy": 0.7,
            "Snowy": 0.5, "Foggy": 0.85
        }
        
        self.time_demand = {
            6: 0.2, 7: 0.4, 8: 0.9, 9: 1.0, 10: 0.95,
            11: 0.8, 12: 0.85, 13: 0.9, 14: 0.85, 15: 0.7,
            16: 0.75, 17: 1.0, 18: 0.9, 19: 0.7, 20: 0.5, 21: 0.3
        }
        
        # Training tracking
        self.days_trained = 0
        self.total_reward = 0
        self.episode_rewards = []
    
    def get_state(self, occupancy, hour, weather):
        """Create discrete state for Q-Learning"""
        # Discretize occupancy (5 levels)
        occ_level = min(4, int(occupancy * 5))
        # Discretize hour (3 periods)
        hour_period = 0 if hour < 12 else (1 if hour < 18 else 2)
        return (occ_level, hour_period, weather)
    
    def get_price_actions(self, occupancy, hour, weather):
        """Generate possible price actions"""
        base_boost = occupancy ** 2 * (self.max_price - self.base_price)
        hour_demand = self.time_demand.get(hour, 0.5)
        weather_mult = self.weather_demand.get(weather, 1.0)
        
        # Generate 5 price candidates
        base_calc = self.base_price + base_boost * hour_demand * weather_mult
        
        actions = [
            max(self.min_price, base_calc * 0.7),   # Low
            max(self.min_price, base_calc * 0.85),  # Medium-Low
            max(self.min_price, base_calc * 1.0),   # Medium
            max(self.min_price, base_calc * 1.2),   # Medium-High
            min(self.max_price, base_calc * 1.4),   # High
        ]
        return [round(p, 2) for p in actions]
    
    def get_q_value(self, state, price_action):
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if price_action not in self.q_table[state]:
            self.q_table[state][price_action] = 0.5
        return self.q_table[state][price_action]
    
    def update_q_value(self, state, price_action, reward, next_state):
        """Update Q-value using Q-Learning formula"""
        current_q = self.get_q_value(state, price_action)
        
        # Get best Q-value for next state
        next_actions = self.time_demand  # Just for valid actions
        next_q_values = []
        if next_state in self.q_table:
            next_q_values = list(self.q_table[next_state].values())
        max_next_q = max(next_q_values) if next_q_values else 0.5
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][price_action] = new_q
        
        # Store rewards for training history
        self.state_rewards[state].append(reward)
    
    def select_price_action(self, state, occupancy, hour, weather, training=True):
        """Epsilon-greedy action selection"""
        possible_prices = self.get_price_actions(occupancy, hour, weather)
        
        # Epsilon-greedy: explore or exploit
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(possible_prices)
        else:
            # Exploitation: best learned action
            best_price = None
            best_q = -float('inf')
            
            for price in possible_prices:
                q_val = self.get_q_value(state, price)
                if q_val > best_q:
                    best_q = q_val
                    best_price = price
            
            return best_price if best_price else possible_prices[2]
    
    def get_optimal_price(self, occupancy, hour, weather, is_weekend, day_num, training=True):
        """
        Main method: Select optimal price using trained Q-Learning policy
        Improves with each day of training
        """
        state = self.get_state(occupancy, hour, weather)
        
        # Select price action using epsilon-greedy
        price = self.select_price_action(state, occupancy, hour, weather, training)
        
        # Decay exploration over time
        if training:
            self.epsilon = max(0.05, self.epsilon * 0.98)
        
        self.price_history.append(price)
        return price
    
    def train_on_day_reward(self, day_states, day_rewards):
        """
        Train Q-Learning model with rewards from a complete day
        Called at end of each day
        """
        total_day_reward = sum(day_rewards)
        self.episode_rewards.append(total_day_reward)
        self.total_reward = total_day_reward
        
        # Update Q-values for all state-action pairs in the day
        for i, state in enumerate(day_states[:-1]):
            price = self.price_history[-(len(day_states)-i)]
            reward = day_rewards[i] if i < len(day_rewards) else 0
            next_state = day_states[i + 1] if i + 1 < len(day_states) else state
            
            self.update_q_value(state, price, reward, next_state)
        
        self.days_trained += 1
        
        # Reduce exploration as we train
        if self.days_trained > 1:
            self.learning_rate = max(0.01, self.learning_rate * 0.99)

 

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
        """Draw car"""
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
        pygame.draw.rect(surface, (200, 200, 200), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(surface, COLORS['parking_line'], (self.x, self.y, self.width, self.height), 2)
        
        if not self.occupied:
            font = pygame.font.Font(None, 12)
            num_text = font.render(str(self.space_id), True, COLORS['text_light'])
            surface.blit(num_text, (self.x + 8, self.y + 10))


class ParkingLot:
    """Complete parking lot"""
    def __init__(self, num_spaces=50):  # Bigger lot - 50 spaces
        self.spaces = []
        self.cars = {}
        self.car_counter = 1000
        self.num_spaces = num_spaces
        
        # Create parking spaces in grid
        start_x = 300
        start_y = 200
        space_spacing_x = 60
        space_spacing_y = 50
        
        space_id = 1
        for row in range(5):
            for col in range(10):
                if space_id <= num_spaces:
                    x = start_x + col * space_spacing_x
                    y = start_y + row * space_spacing_y
                    self.spaces.append(ParkingSpace(x, y, space_id))
                    space_id += 1
    
    def get_occupancy(self):
        occupied = sum(1 for space in self.spaces if space.occupied)
        return occupied / len(self.spaces)
    
    def get_free_spaces(self):
        return [s for s in self.spaces if not s.occupied]
    
    def add_car(self, car):
        free = self.get_free_spaces()
        if free:
            space = free[0]
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


 
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
    
    def draw(self, surface, font):
        color = COLORS['button_hover'] if self.hovered else COLORS['button']
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 2)
        
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


 
class RLCityParkingSimulator:
    """Main simulator with RL pricing and city parking patterns"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🅿️ CITY PARKING LOT - RL Pricing Model")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Initialize RL model (use A2C if available, fallback to Q-Learning)
        if A2C_AVAILABLE:
            self.rl_model = A2CPricingModel()
            self.using_a2c = True
            print("✅ Using A2C Agent for pricing")
        else:
            self.rl_model = RLPricingModel()
            self.using_a2c = False
            print("📚 Using Q-Learning Model (A2C not available)")
        
        # Components (50-space lot for realistic city parking)
        self.parking_lot = ParkingLot(num_spaces=50)
        
        # Day management
        self.current_day_num = 1
        self.current_date = datetime.now().replace(hour=6, minute=0, second=0)
        self.day_start_time = self.current_date
        self.day_end_time = self.current_date.replace(hour=18)
        self.current_time = self.day_start_time
        
        # Day status
        self.day_complete = False
        self.day_revenue = 0.0
        self.day_cars = 0
        self.previous_day_revenue = 0.0
        self.previous_day_cars = 0
        self.prev_day_avg_price = 0.0
        self.prev_day_avg_occupancy = 0.0
        
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
        self.occupancy_history = []
        self.last_add_time = 0
        self.add_interval = 0.5
        self.simulation_speed = 3.0
        
        # Training tracking for RL
        self.day_states = []
        self.day_rewards = []
        
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
    
    def maintain_minimum_occupancy(self):
        """Ensure lot maintains 60% minimum occupancy (realistic city parking)"""
        target_occupancy = 0.60
        current_occupancy = self.parking_lot.get_occupancy()
        
        # If below target, add more cars
        while current_occupancy < target_occupancy:
            hour = self.current_time.hour
            is_weekend = self.current_time.weekday() >= 5
            
            price = self.rl_model.get_optimal_price(
                current_occupancy, hour, self.current_weather, is_weekend, self.current_day_num
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
            
            current_occupancy = self.parking_lot.get_occupancy()
    
    def add_cars(self):
        """Add cars based on RL model predictions"""
        hour = self.current_time.hour
        occupancy = self.parking_lot.get_occupancy()
        is_weekend = self.current_time.weekday() >= 5
        
        # Get state for RL training
        state = self.rl_model.get_state(occupancy, hour, self.current_weather)
        self.day_states.append(state)
        
        # RL Model decides how many cars based on state
        base_arrivals = 2 if occupancy < 0.9 else 1
        
        # Adjust based on time demand
        hour_factor = self.rl_model.time_demand.get(hour, 0.5)
        num_arrivals = int(base_arrivals * hour_factor * 2)
        
        # Weather effect
        weather_multiplier = self.rl_model.weather_demand.get(self.current_weather, 1.0)
        num_arrivals = int(num_arrivals * weather_multiplier)
        
        num_arrivals = min(num_arrivals, len(self.parking_lot.get_free_spaces()))
        
        # Add cars
        for _ in range(num_arrivals):
            occupancy = self.parking_lot.get_occupancy()
            
            # RL pricing decision - WITH TRAINING
            price = self.rl_model.get_optimal_price(
                occupancy, hour, self.current_weather, is_weekend, self.current_day_num,
                training=True  # Enable RL training
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
                reward = price  # Reward is the price/revenue
                self.day_rewards.append(reward)
                self.day_revenue += price
                self.total_revenue += price
                self.day_cars += 1
                self.total_cars += 1
                self.prices_history.append(price)
    
    def remove_expired_cars(self):
        """Remove parked cars that have overstayed"""
        cars_to_remove = []
        for car_id, car in self.parking_lot.cars.items():
            age_hours = (self.current_time - car.entry_time).total_seconds() / 3600
            if age_hours >= car.duration:
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            self.parking_lot.remove_car(car_id)
    
    def start_next_day(self):
        """Progress to next day and train RL model"""
        # TRAIN the model on today's experience
        if self.day_states and self.day_rewards:
            self.rl_model.train_on_day_reward(self.day_states, self.day_rewards)
        
        # Save stats
        self.previous_day_revenue = self.day_revenue
        self.previous_day_cars = self.day_cars
        if self.day_cars > 0:
            self.prev_day_avg_price = self.day_revenue / self.day_cars
        
        if self.occupancy_history:
            self.prev_day_avg_occupancy = np.mean(self.occupancy_history)
        
        # Reset for new day
        self.current_day_num += 1
        self.current_date += timedelta(days=1)
        self.day_start_time = self.current_date.replace(hour=6)
        self.day_end_time = self.current_date.replace(hour=18)
        self.current_time = self.day_start_time
        self.day_revenue = 0.0
        self.day_cars = 0
        self.day_complete = False
        self.parking_lot = ParkingLot(num_spaces=50)
        self.current_weather = random.choice(self.weather_list)
        self.paused = False
        self.occupancy_history = []
        
        # Reset day tracking
        self.day_states = []
        self.day_rewards = []
    
    def handle_events(self):
        """Handle input"""
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
        
        # Maintain minimum occupancy
        self.maintain_minimum_occupancy()
        
        # Advance time
        self.current_time += timedelta(minutes=self.simulation_speed * dt * 60)
        
        # Track occupancy
        self.occupancy_history.append(self.parking_lot.get_occupancy())
        
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
        
        self.draw_title_bar()
        self.draw_lot_background()
        self.parking_lot.draw(self.screen)
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_previous_day_corner()
        self.draw_rl_info()
        
        if self.hovered_car:
            self.draw_car_tooltip()
        
        self.draw_bottom_bar()
        self.next_day_button.draw(self.screen, self.font_button)
        
        if self.day_complete:
            self.draw_day_complete()
        
        if self.paused and not self.day_complete:
            pause_text = self.font_header.render("⏸ PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_text, (WIDTH - 300, 85))
        
        pygame.display.flip()
    
    def draw_title_bar(self):
        title = self.font_title.render(f"🅿️ CITY PARKING LOT - Day {self.current_day_num} (RL Pricing)", True, COLORS['text'])
        self.screen.blit(title, (30, 20))
    
    def draw_lot_background(self):
        pygame.draw.rect(self.screen, COLORS['asphalt'], (280, 180, 800, 400))
        pygame.draw.rect(self.screen, (100, 100, 100), (280, 180, 800, 400), 3)
        
        pygame.draw.rect(self.screen, COLORS['gate'], (270, 300, 15, 50))
        entrance_text = self.font_small.render("IN", True, (255, 255, 255))
        self.screen.blit(entrance_text, (275, 310))
        
        pygame.draw.rect(self.screen, COLORS['gate'], (1095, 300, 15, 50))
        exit_text = self.font_small.render("OUT", True, (255, 255, 255))
        self.screen.blit(exit_text, (1095, 310))
    
    def draw_left_panel(self):
        x, y = 30, 180
        
        title = self.font_header.render("📅 TODAY'S CONDITIONS", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        time_str = self.current_time.strftime("%H:%M")
        day_str = self.current_date.strftime("%A, %b %d")
        occupancy = self.parking_lot.get_occupancy()
        
        conditions = [
            f"⏰ Time: {time_str}",
            f"📅 Date: {day_str}",
            f"🌦️ {self.weather_emoji[self.current_weather]} {self.current_weather}",
            f"",
            f"📊 Lot Status:",
            f"   {len(self.parking_lot.cars)}/50 spaces ({occupancy*100:.1f}%)",
            f"   (Min: 60% maintained)",
            f"",
            f"💰 Today: ${self.day_revenue:,.2f}",
            f"🚗 Cars: {self.day_cars}",
        ]
        
        for condition in conditions:
            if condition == "":
                y += 10
            else:
                color = (100, 200, 100) if "Today" in condition or "Min" in condition else COLORS['text']
                text = self.font_small.render(condition, True, color)
                self.screen.blit(text, (x, y))
            y += 25
    
    def draw_right_panel(self):
        x, y = WIDTH - 320, 180
        
        title = self.font_header.render("📊 CUMULATIVE STATS", True, COLORS['text'])
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
    
    def draw_rl_info(self):
        """Draw RL pricing information and training progress"""
        x, y = WIDTH - 320, HEIGHT - 280
        
        # Background
        pygame.draw.rect(self.screen, (230, 220, 255), (x, y, 300, 200))
        pygame.draw.rect(self.screen, COLORS['rl_accent'], (x, y, 300, 200), 2)
        
        title = self.font_header.render("🤖 RL PRICING MODEL", True, COLORS['rl_accent'])
        self.screen.blit(title, (x + 10, y + 10))
        
        y += 45
        
        # Current RL decision
        hour = self.current_time.hour
        is_weekend = self.current_date.weekday() >= 5
        occupancy = self.parking_lot.get_occupancy()
        
        current_price = self.rl_model.get_optimal_price(
            occupancy, hour, self.current_weather, is_weekend, self.current_day_num,
            training=False  # Don't train on display call
        )
        
        # State representation
        state_info = [
            f"State: Occ={occupancy*100:.0f}%",
            f"       Time={hour}:00",
            f"",
            f"Price: ${current_price:.2f}",
            f"Learning Rate: {self.rl_model.learning_rate:.3f}",
            f"Exploration: {self.rl_model.epsilon:.2f}",
        ]
        
        for line in state_info:
            if line == "":
                y += 8
            else:
                text = self.font_small.render(line, True, COLORS['text'])
                self.screen.blit(text, (x + 10, y))
            y += 22
        
        # Training progress
        y += 5
        trained_days = self.rl_model.days_trained
        trained_text = self.font_small.render(f"Trained on {trained_days} days", True, (100, 200, 100))
        self.screen.blit(trained_text, (x + 10, y))
    
    def draw_previous_day_corner(self):
        x, y = 30, HEIGHT - 180
        
        pygame.draw.rect(self.screen, (240, 240, 255), (x, y, 280, 160))
        pygame.draw.rect(self.screen, (100, 100, 200), (x, y, 280, 160), 3)
        
        title = self.font_header.render("📈 PREVIOUS DAY", True, (50, 50, 150))
        self.screen.blit(title, (x + 10, y + 10))
        
        y += 45
        
        if self.current_day_num > 1:
            stats = [
                f"Revenue: ${self.previous_day_revenue:,.2f}",
                f"Cars: {self.previous_day_cars}",
                f"Avg Price: ${self.prev_day_avg_price:.2f}",
                f"Occupancy: {self.prev_day_avg_occupancy*100:.0f}%",
            ]
        else:
            stats = [
                f"Day 1 - No previous data",
                f"",
                f"",
                f"",
            ]
        
        for stat in stats:
            if stat:
                text = self.font_normal.render(stat, True, COLORS['text'])
                self.screen.blit(text, (x + 15, y))
            y += 30
    
    def draw_car_tooltip(self):
        car = self.hovered_car
        x, y = self.mouse_pos
        
        font_small = pygame.font.Font(None, 16)
        lines = [
            f"Car #{car.car_id}",
            f"Price: ${car.price:.2f}",
            f"Duration: {car.duration:.1f}h"
        ]
        
        box_x = min(x + 10, WIDTH - 140)
        box_y = min(y + 10, HEIGHT - 70)
        
        pygame.draw.rect(self.screen, (40, 40, 40), (box_x, box_y, 140, 70))
        pygame.draw.rect(self.screen, (255, 255, 100), (box_x, box_y, 140, 70), 2)
        
        for i, line in enumerate(lines):
            text = font_small.render(line, True, (255, 255, 255))
            self.screen.blit(text, (box_x + 8, box_y + 5 + i*20))
    
    def draw_bottom_bar(self):
        y = HEIGHT - 70
        
        pygame.draw.rect(self.screen, (200, 200, 200), (0, y, WIDTH, 70))
        pygame.draw.line(self.screen, (100, 100, 100), (0, y), (WIDTH, y), 2)
        
        if self.day_complete:
            controls_text = "✅ DAY COMPLETE! Click 'NEXT DAY' to continue"
        else:
            controls_text = "CONTROLS: SPACE=Pause | ESC=Exit"
        
        controls = self.font_small.render(controls_text, True, COLORS['text'])
        self.screen.blit(controls, (30, y + 15))
        
        instr_text = "💡 RL Model maintains 60% minimum occupancy | Watch dynamic pricing in action!"
        instr = self.font_small.render(instr_text, True, COLORS['text_light'])
        self.screen.blit(instr, (30, y + 40))
    
    def draw_day_complete(self):
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        complete_text = self.font_title.render("✅ DAY COMPLETE!", True, (100, 255, 100))
        complete_rect = complete_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        self.screen.blit(complete_text, complete_rect)
        
        stats_text = self.font_header.render(
            f"Revenue: ${self.day_revenue:,.2f} | Cars: {self.day_cars} | Avg Price: ${self.day_revenue/self.day_cars if self.day_cars > 0 else 0:.2f}",
            True, (255, 255, 100)
        )
        stats_rect = stats_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(stats_text, stats_rect)
    
    def run(self):
        print("\n" + "="*80)
        print("🅿️ CITY PARKING LOT SIMULATOR - RL PRICING MODEL")
        print("="*80)
        print("\nFeatures:")
        print("  ✓ 50-space realistic city parking lot")
        print("  ✓ 60% minimum occupancy (maintained by RL model)")
        print("  ✓ RL-based dynamic pricing")
        print("  ✓ State-action-reward learning simulation")
        print("  ✓ Day-by-day progression with comparison")
        print("\nClick NEXT DAY to progress, watch how RL prices respond!")
        print("="*80 + "\n")
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        print(f"Days Simulated: {self.current_day_num}")
        print(f"Total Revenue: ${self.total_revenue:,.2f}")
        print(f"Total Cars: {self.total_cars}")
        if self.prices_history:
            print(f"Avg Price: ${np.mean(self.prices_history):.2f}")
            print(f"Price Range: ${min(self.prices_history):.2f} - ${max(self.prices_history):.2f}")
        print("="*80 + "\n")
        
        pygame.quit()


if __name__ == "__main__":
    simulator = RLCityParkingSimulator()
    simulator.run()
