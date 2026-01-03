"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║         🚗 DYNAMIC PARKING PRICING - PREMIUM AI DASHBOARD 2.0 🚗          ║
║                                                                           ║
║   Modern Real-Time Visualization for A2C Reinforcement Learning Agent    ║
║         With Advanced Analytics, Real Parking Simulation & Design        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pygame
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import sys
import torch
import json

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from role_1.env import ParkingPricingEnv
    from role_2.a2c_new import A2CAgent, A2CConfig
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import environment/agent: {e}")
    ENV_AVAILABLE = False

pygame.init()

# ═══════════════════════════════════════════════════════════════════════════
# MODERN COLOR PALETTE - GRADIENT BASED DESIGN
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    # Base colors
    'bg_dark': (8, 13, 27),          # Deep navy
    'bg_light': (13, 20, 35),        # Slightly lighter navy
    'surface': (20, 30, 50),         # Card surface
    'surface_elevated': (25, 35, 60),  # Elevated cards
    
    # Accent colors
    'primary': (100, 200, 255),      # Cyan-blue
    'primary_dark': (70, 140, 200),  # Dark cyan
    'primary_light': (150, 220, 255),  # Light cyan
    
    'success': (76, 180, 80),        # Green
    'warning': (255, 165, 50),       # Orange
    'danger': (255, 80, 90),         # Red
    'accent': (180, 100, 255),       # Purple
    
    # Text
    'text_primary': (240, 245, 250),   # Near white
    'text_secondary': (160, 175, 195), # Gray
    'text_muted': (100, 120, 150),     # Darker gray
    
    # Special
    'success_light': (120, 220, 140),
    'warning_light': (255, 200, 100),
    'danger_light': (255, 150, 150),
    'border': (50, 80, 130),
    'border_light': (80, 120, 180),
}

# ═══════════════════════════════════════════════════════════════════════════
# FONTS
# ═══════════════════════════════════════════════════════════════════════════

FONT_TITLE = pygame.font.Font(None, 56)
FONT_HEADER = pygame.font.Font(None, 40)
FONT_LARGE = pygame.font.Font(None, 32)
FONT_MEDIUM = pygame.font.Font(None, 24)
FONT_SMALL = pygame.font.Font(None, 18)
FONT_TINY = pygame.font.Font(None, 14)

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1440
FPS = 30
PARKING_SPACES = 150

class ModernSimulator:
    """Realistic parking simulator with actual car arrivals/departures"""
    
    def __init__(self, capacity=150):
        self.env = ParkingPricingEnv()
        self.agent = None
        self.capacity = capacity
        self.cars_parked = 0
        self.current_price = 5.0
        self.total_revenue = 0
        self.episode = 0
        self.current_step = 0
        
        # Initialize agent if available
        try:
            model_path = PROJECT_ROOT / "training_results" / "a2c_best" / "best_model_ep84.pth"
            if model_path.exists():
                config = A2CConfig(state_dim=5, action_dim=1, hidden_dim=256)
                self.agent = A2CAgent(config)
                try:
                    # Load with weights_only to avoid pickle issues
                    state = torch.load(str(model_path), map_location='cpu', weights_only=False)
                    if isinstance(state, dict):
                        self.agent.policy_net.load_state_dict(state.get('policy_net', {}))
                        self.agent.value_net.load_state_dict(state.get('value_net', {}))
                    self.agent_available = True
                    print("✓ Trained agent loaded successfully")
                except Exception as load_err:
                    print(f"⚠ Could not load weights: {load_err}, using heuristic")
                    self.agent_available = False
            else:
                self.agent_available = False
                print("⚠ Trained agent not found, using heuristic pricing")
        except Exception as e:
            print(f"⚠ Agent initialization: {e}")
            self.agent_available = False
        
        # History tracking
        self.price_history = deque(maxlen=288)  # 24 hours
        self.occupancy_history = deque(maxlen=288)
        self.revenue_history = deque(maxlen=288)
        self.car_entries = deque(maxlen=50)  # Recent car entries
        self.car_exits = deque(maxlen=50)   # Recent car exits
        self.episode_rewards = deque(maxlen=1000)
        
        # Demand curve (more realistic)
        self.hourly_base_demand = self._generate_demand_curve()
        self.step_base_demand = np.repeat(self.hourly_base_demand, 12)  # 5-min steps
        
        # Reset environment
        self.state, _ = self.env.reset()
        self.current_reward = 0
        self.cumulative_reward = 0
        
    def _generate_demand_curve(self):
        """Generate realistic hourly demand pattern"""
        hours = np.arange(24)
        # Peak at 8am and 5pm (work hours)
        demand = 0.3 + 0.4 * np.sin((hours - 8) * np.pi / 12) + \
                 0.3 * np.sin((hours - 17) * np.pi / 12)
        return np.clip(demand, 0.1, 1.0)
    
    def step(self):
        """Execute one simulation step (5 minutes)"""
        self.current_step += 1
        
        # Get next price from agent or random
        if self.agent and self.agent_available:
            with torch.no_grad():
                action, _, _ = self.agent.select_action(
                    torch.FloatTensor(self.state).unsqueeze(0),
                    training=False
                )
            self.current_price = float(np.clip(action[0, 0], 1.5, 25.0))
        else:
            # Simple heuristic: lower price when occupancy low, raise when high
            occupancy = self.state[0]
            if occupancy < 0.6:
                self.current_price = max(2.0, self.current_price - 0.5)
            elif occupancy > 0.85:
                self.current_price = min(25.0, self.current_price + 1.0)
        
        # Step environment
        self.state, reward, terminated, truncated, info = self.env.step(
            np.array([self.current_price])
        )
        
        self.current_reward = float(reward)
        self.cumulative_reward += reward
        
        # Update parking occupancy based on environment
        self.cars_parked = int(self.state[0] * self.capacity)
        
        # Track revenue
        self.total_revenue += max(0, reward / 10)  # Normalize reward to revenue
        
        # Record history
        self.price_history.append(self.current_price)
        self.occupancy_history.append(self.state[0] * 100)
        self.revenue_history.append(reward / 10)
        
        # Check for episode end
        if terminated or truncated or self.current_step >= 288:
            self.episode_rewards.append(self.cumulative_reward)
            self.episode += 1
            self.current_step = 0
            self.cumulative_reward = 0
            self.state, _ = self.env.reset()
    
    def get_stats(self):
        """Get current statistics"""
        occupancy_pct = (self.cars_parked / self.capacity) * 100
        
        # Calculate averages
        avg_price = np.mean(self.price_history) if self.price_history else self.current_price
        avg_occupancy = np.mean(self.occupancy_history) if self.occupancy_history else occupancy_pct
        price_std = np.std(self.price_history) if len(self.price_history) > 1 else 0
        
        # Episode stats
        avg_episode_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        return {
            'cars_parked': self.cars_parked,
            'capacity': self.capacity,
            'occupancy_pct': occupancy_pct,
            'current_price': self.current_price,
            'avg_price': avg_price,
            'price_std': price_std,
            'total_revenue': self.total_revenue,
            'current_reward': self.current_reward,
            'cumulative_reward': self.cumulative_reward,
            'episode': self.episode,
            'step': self.current_step,
            'avg_occupancy': avg_occupancy,
            'avg_episode_reward': avg_episode_reward,
            'agent_available': self.agent_available,
        }


class UIRenderer:
    """Modern UI rendering system with gradient effects"""
    
    def __init__(self, screen):
        self.screen = screen
        self.time_start = pygame.time.get_ticks()
    
    def draw_gradient_rect(self, x, y, w, h, color1, color2, vertical=True):
        """Draw a gradient rectangle"""
        surface = pygame.Surface((w, h))
        
        if vertical:
            for i in range(h):
                ratio = i / h
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surface, (r, g, b), (0, i), (w, i))
        else:
            for i in range(w):
                ratio = i / w
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surface, (r, g, b), (i, 0), (i, h))
        
        self.screen.blit(surface, (x, y))
    
    def draw_rounded_rect(self, x, y, w, h, color, radius=12, border_color=None, border_width=2):
        """Draw a rounded rectangle with optional border"""
        # Create surface for rounded rect
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surface, color, (0, 0, w, h), border_radius=radius)
        
        if border_color:
            pygame.draw.rect(surface, border_color, (0, 0, w, h), border_width, border_radius=radius)
        
        self.screen.blit(surface, (x, y))
    
    def draw_card(self, x, y, w, h, title="", gradient=True):
        """Draw a modern card with rounded corners and gradient"""
        # Background gradient
        if gradient:
            self.draw_gradient_rect(x, y, w, h, COLORS['surface'], COLORS['surface_elevated'])
        else:
            self.draw_rounded_rect(x, y, w, h, COLORS['surface'])
        
        # Border
        self.draw_rounded_rect(x, y, w, h, COLORS['border_light'], border_color=COLORS['border'], border_width=1)
        
        # Title if provided
        if title:
            title_text = FONT_SMALL.render(title, True, COLORS['primary'])
            self.screen.blit(title_text, (x + 20, y + 12))
    
    def draw_metric_big(self, x, y, w, h, icon, label, value, unit="", color=None):
        """Draw a large metric card with icon"""
        color = color or COLORS['primary']
        
        self.draw_card(x, y, w, h)
        
        # Icon
        icon_text = FONT_LARGE.render(icon, True, COLORS['text_primary'])
        self.screen.blit(icon_text, (x + 20, y + 20))
        
        # Label
        label_text = FONT_SMALL.render(label, True, COLORS['text_secondary'])
        self.screen.blit(label_text, (x + 20, y + 65))
        
        # Value with color
        value_str = f"{value:.1f}" if isinstance(value, float) else str(value)
        value_text = FONT_HEADER.render(value_str, True, color)
        self.screen.blit(value_text, (x + 20, y + 95))
        
        # Unit
        if unit:
            unit_text = FONT_SMALL.render(unit, True, COLORS['text_muted'])
            self.screen.blit(unit_text, (x + 20 + value_text.get_width() + 10, y + 110))
    
    def draw_progress_circular(self, x, y, radius, percent, color):
        """Draw a circular progress indicator"""
        # Background circle
        pygame.draw.circle(self.screen, COLORS['border'], (x, y), radius, 3)
        
        # Progress arc
        if percent > 0:
            arc_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
            angle = (percent / 100) * 360
            pygame.draw.arc(self.screen, color, arc_rect, 0, np.deg2rad(angle), 4)
        
        # Center text
        percent_text = FONT_MEDIUM.render(f"{percent:.0f}%", True, COLORS['text_primary'])
        text_rect = percent_text.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(percent_text, text_rect)
    
    def draw_graph(self, x, y, w, h, data, title="", max_val=None):
        """Draw a simple graph visualization"""
        self.draw_card(x, y, w, h, title)
        
        if not data or len(data) < 2:
            return
        
        # Calculate max value
        data_array = np.array(list(data), dtype=float)
        if max_val is None:
            max_val = float(np.max(data_array)) if len(data_array) > 0 else 1.0
        
        if max_val <= 0:
            max_val = 1.0
        
        # Draw grid
        grid_spacing = w // 10
        for i in range(10):
            grid_x = x + 30 + i * grid_spacing
            pygame.draw.line(self.screen, COLORS['border'], (grid_x, y + 40), (grid_x, y + h - 20), 1)
        
        # Draw data line
        data_len = len(data)
        point_spacing = (w - 60) / max(data_len - 1, 1)
        points = []
        
        for i, val in enumerate(data):
            try:
                val = float(val)
                px = x + 30 + i * point_spacing
                py = y + h - 20 - (val / max_val) * (h - 60)
                points.append((int(px), int(py)))
            except (ValueError, TypeError):
                continue
        
        # Draw line only if we have valid points
        if len(points) >= 2:
            pygame.draw.lines(self.screen, COLORS['primary'], points, 2)
        
        # Draw points - last 5 points
        for point in points[-5:]:
            pygame.draw.circle(self.screen, COLORS['primary_light'], point, 4)


class ModernDashboard:
    """Main dashboard application with modern design"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🚗 Dynamic Parking Pricing - Premium Dashboard")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        print("\n" + "="*80)
        print("MODERN DASHBOARD INITIALIZATION".center(80))
        print("="*80)
        
        try:
            self.simulator = ModernSimulator(PARKING_SPACES)
            self.renderer = UIRenderer(self.screen)
            
            print(f"✓ Environment: Loaded (150 spaces)")
            agent_status = "✓ A2C Agent (Trained Model)" if self.simulator.agent_available else "⚠ Heuristic Pricing"
            print(f"{agent_status}")
            print(f"✓ Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
            print("="*80)
            print("Controls: SPACE=Pause | ESC=Quit | 'S'=Save Stats | 'R'=Reset")
            print("="*80 + "\n")
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RUNNING"
                    print(f"⏸  Dashboard {status}")
                elif event.key == pygame.K_s:
                    self.save_stats()
                elif event.key == pygame.K_r:
                    self.simulator = ModernSimulator(PARKING_SPACES)
                    print("🔄 Dashboard reset")
    
    def save_stats(self):
        """Save current statistics to file"""
        stats = self.simulator.get_stats()
        timestamp = datetime.now().isoformat()
        data = {'timestamp': timestamp, 'stats': stats}
        
        output_file = PROJECT_ROOT / "dashboard_data.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Stats saved to {output_file}")
    
    def render(self):
        """Render the complete dashboard"""
        # Clear screen
        self.screen.fill(COLORS['bg_dark'])
        
        # Get stats
        stats = self.simulator.get_stats()
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        
        title = FONT_TITLE.render("🚗 DYNAMIC PARKING PRICING SYSTEM", True, COLORS['primary'])
        self.screen.blit(title, (40, 20))
        
        time_str = datetime.now().strftime("%H:%M:%S")
        time_text = FONT_SMALL.render(f"Time: {time_str} | Status: {'PAUSED' if self.paused else 'RUNNING'}", 
                                     True, COLORS['text_secondary'])
        self.screen.blit(time_text, (40, 70))
        
        # ═══════════════════════════════════════════════════════════════════
        # TOP ROW: KEY METRICS
        # ═══════════════════════════════════════════════════════════════════
        
        # Occupancy (with circular progress)
        self.renderer.draw_card(40, 120, 380, 240)
        occ_color = COLORS['danger'] if stats['occupancy_pct'] > 85 else \
                   COLORS['warning'] if stats['occupancy_pct'] > 70 else COLORS['success']
        
        label = FONT_MEDIUM.render("OCCUPANCY", True, COLORS['text_secondary'])
        self.screen.blit(label, (60, 130))
        self.renderer.draw_progress_circular(230, 200, 60, stats['occupancy_pct'], occ_color)
        
        cars_text = FONT_SMALL.render(f"{stats['cars_parked']}/{stats['capacity']} cars", 
                                     True, COLORS['text_muted'])
        self.screen.blit(cars_text, (130, 285))
        
        # Price
        price_color = COLORS['danger'] if stats['current_price'] > 15 else \
                     COLORS['warning'] if stats['current_price'] > 10 else COLORS['success']
        self.renderer.draw_metric_big(440, 120, 380, 240, "💵", "CURRENT PRICE", 
                                     stats['current_price'], "$/hour", price_color)
        
        # Revenue
        self.renderer.draw_metric_big(840, 120, 380, 240, "💰", "TOTAL REVENUE",
                                     stats['total_revenue'], "$", COLORS['warning_light'])
        
        # Episode
        self.renderer.draw_metric_big(1240, 120, 380, 240, "🎯", "EPISODE",
                                     stats['episode'], "", COLORS['accent'])
        
        # Reward
        self.renderer.draw_metric_big(1640, 120, 380, 240, "⭐", "CURR REWARD",
                                     stats['current_reward'], "$", COLORS['success_light'])
        
        # Agent Status
        status_text = "✓ AI AGENT" if stats['agent_available'] else "⚠ HEURISTIC"
        status_color = COLORS['success'] if stats['agent_available'] else COLORS['warning']
        self.renderer.draw_card(2040, 120, 480, 240)
        agent_label = FONT_MEDIUM.render("PRICING METHOD", True, COLORS['text_secondary'])
        self.screen.blit(agent_label, (2060, 130))
        agent_text = FONT_LARGE.render(status_text, True, status_color)
        self.screen.blit(agent_text, (2060, 180))
        algo_text = FONT_SMALL.render("A2C Actor-Critic", True, COLORS['text_muted'])
        self.screen.blit(algo_text, (2060, 240))
        
        # ═══════════════════════════════════════════════════════════════════
        # MIDDLE ROW: GRAPHS
        # ═══════════════════════════════════════════════════════════════════
        
        self.renderer.draw_graph(40, 390, 1200, 280, self.simulator.price_history, 
                               "📈 PRICE HISTORY (Last 24 Hours)", max_val=25)
        
        self.renderer.draw_graph(1260, 390, 1260, 280, self.simulator.occupancy_history,
                               "📊 OCCUPANCY TRENDS", max_val=100)
        
        # ═══════════════════════════════════════════════════════════════════
        # BOTTOM ROW: DETAILED STATS & LEARNING
        # ═══════════════════════════════════════════════════════════════════
        
        # Detailed metrics
        self.renderer.draw_card(40, 700, 900, 300)
        stats_title = FONT_MEDIUM.render("DETAILED METRICS", True, COLORS['primary'])
        self.screen.blit(stats_title, (60, 715))
        
        metrics_list = [
            ("Average Price", f"${stats['avg_price']:.2f}/hour", COLORS['text_secondary']),
            ("Avg Occupancy", f"{stats['avg_occupancy']:.1f}%", COLORS['text_secondary']),
            ("Price Stability", f"σ = {stats['price_std']:.2f}", COLORS['text_secondary']),
            ("Target Occupancy", "80%", COLORS['success']),
            ("Current Step", f"{stats['step']}/288", COLORS['text_secondary']),
        ]
        
        for i, (metric, value, color) in enumerate(metrics_list):
            metric_text = FONT_SMALL.render(f"{metric}: ", True, COLORS['text_secondary'])
            value_text = FONT_SMALL.render(value, True, color)
            y_pos = 760 + i * 45
            self.screen.blit(metric_text, (60, y_pos))
            self.screen.blit(value_text, (350, y_pos))
        
        # Learning performance
        self.renderer.draw_card(960, 700, 1540, 300)
        learning_title = FONT_MEDIUM.render("A2C LEARNING PERFORMANCE", True, COLORS['primary'])
        self.screen.blit(learning_title, (980, 715))
        
        learning_list = [
            ("Avg Episode Reward", f"${stats['avg_episode_reward']:.2f}", COLORS['success_light']),
            ("Episodes Completed", f"{stats['episode']} episodes", COLORS['text_secondary']),
            ("Current Cumulative", f"${stats['cumulative_reward']:.2f}", COLORS['accent']),
            ("Algorithm", "Actor-Critic (A2C) from Scratch", COLORS['primary']),
            ("Network Size", "5→256→256→1 (Policy & Value)", COLORS['text_secondary']),
        ]
        
        for i, (metric, value, color) in enumerate(learning_list):
            metric_text = FONT_SMALL.render(f"{metric}: ", True, COLORS['text_secondary'])
            value_text = FONT_SMALL.render(value, True, color)
            y_pos = 760 + i * 45
            self.screen.blit(metric_text, (980, y_pos))
            self.screen.blit(value_text, (1280, y_pos))
        
        # ═══════════════════════════════════════════════════════════════════
        # BOTTOM: CONTROLS & INFO
        # ═══════════════════════════════════════════════════════════════════
        
        controls_text = "SPACE: Pause/Resume  |  S: Save Stats  |  R: Reset  |  ESC: Exit"
        controls = FONT_TINY.render(controls_text, True, COLORS['text_secondary'])
        self.screen.blit(controls, (40, SCREEN_HEIGHT - 30))
        
        # Refresh rate
        fps_text = FONT_TINY.render(f"FPS: {int(self.clock.get_fps())}", True, COLORS['text_muted'])
        self.screen.blit(fps_text, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop"""
        try:
            while self.running:
                self.handle_events()
                
                if not self.paused:
                    try:
                        self.simulator.step()
                    except Exception as step_err:
                        print(f"⚠ Step error: {step_err}")
                        # Continue despite step errors
                        continue
                
                try:
                    self.render()
                except Exception as render_err:
                    print(f"⚠ Render error: {render_err}")
                    # Try to recover from render errors
                    continue
                
                self.clock.tick(FPS)
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Critical error: {e}")
            import traceback
            traceback.print_exc()
            input("Press Enter to close...")  # Keep window open to see error
        finally:
            pygame.quit()
            print("\n✓ Dashboard closed successfully\n")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        dashboard = ModernDashboard()
        dashboard.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
