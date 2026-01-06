"""
COMPREHENSIVE PROJECT DASHBOARD
 
Real-time monitoring and visualization of:
- Parking lot pricing dynamics
- Revenue trends
- Occupancy patterns
- RL Model performance
- All metrics and analytics
Perfect for project tracking!
"""

import pygame
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

pygame.init()

 
WIDTH, HEIGHT = 1600, 1200
FPS = 30

COLORS = {
    'bg': (20, 20, 40),
    'card_bg': (40, 40, 60),
    'card_border': (100, 100, 150),
    'text': (240, 240, 250),
    'text_dim': (150, 150, 180),
    'accent': (100, 200, 255),
    'success': (76, 175, 80),
    'warning': (255, 152, 0),
    'danger': (244, 67, 54),
    'rl_accent': (147, 51, 234),
}

 
class MetricsTracker:
    """Track all project metrics in real-time"""
    
    def __init__(self, max_history=1000):
        # Time series data
        self.timestamps = deque(maxlen=max_history)
        self.prices = deque(maxlen=max_history)
        self.revenues = deque(maxlen=max_history)
        self.occupancies = deque(maxlen=max_history)
        self.cars_count = deque(maxlen=max_history)
        
        # Daily aggregates
        self.daily_revenues = []
        self.daily_cars = []
        self.daily_avg_prices = []
        self.daily_occupancies = []
        
        # Summary stats
        self.total_revenue = 0.0
        self.total_cars = 0
        self.days_simulated = 0
        self.model_training_days = 0
        
        # Current state
        self.current_price = 0.0
        self.current_occupancy = 0.0
        self.current_hour = 6
        self.current_weather = "Sunny"
        self.learning_rate = 0.1
        self.epsilon = 0.2
    
    def add_price_event(self, price, occupancy, timestamp=None):
        """Add price and occupancy data point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.timestamps.append(timestamp)
        self.prices.append(price)
        self.occupancies.append(occupancy)
        self.current_price = price
        self.current_occupancy = occupancy
    
    def add_revenue_event(self, revenue):
        """Add revenue data point"""
        self.revenues.append(revenue)
        self.total_revenue += revenue
    
    def add_car_event(self):
        """Record car arrival"""
        self.total_cars += 1
        self.cars_count.append(self.total_cars)
    
    def end_day(self, day_revenue, day_cars, avg_price, avg_occupancy):
        """Record end of day stats"""
        self.daily_revenues.append(day_revenue)
        self.daily_cars.append(day_cars)
        self.daily_avg_prices.append(avg_price)
        self.daily_occupancies.append(avg_occupancy)
        self.days_simulated += 1
    
    def update_rl_status(self, learning_rate, epsilon, trained_days):
        """Update RL training status"""
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.model_training_days = trained_days
    
    def get_statistics(self):
        """Calculate current statistics"""
        stats = {
            'avg_price': np.mean(self.prices) if self.prices else 0,
            'max_price': max(self.prices) if self.prices else 0,
            'min_price': min(self.prices) if self.prices else 0,
            'avg_occupancy': np.mean(self.occupancies) if self.occupancies else 0,
            'total_revenue': self.total_revenue,
            'total_cars': self.total_cars,
            'days_simulated': self.days_simulated,
            'avg_daily_revenue': np.mean(self.daily_revenues) if self.daily_revenues else 0,
            'revenue_trend': self.get_trend(self.daily_revenues),
            'price_trend': self.get_trend(list(self.prices)[-50:] if len(self.prices) > 50 else list(self.prices)),
        }
        return stats
    
    def get_trend(self, data):
        """Calculate trend direction"""
        if len(data) < 2:
            return 0
        return data[-1] - data[0]


 
class ChartGenerator:
    """Generate matplotlib charts"""
    
    @staticmethod
    def create_price_chart(tracker):
        """Create price trend chart"""
        fig = Figure(figsize=(4, 2), dpi=100, facecolor='#282828')
        ax = fig.add_subplot(111, facecolor='#1a1a2e')
        
        if len(tracker.prices) > 0:
            x = np.arange(len(tracker.prices))[-50:]  # Last 50
            y = list(tracker.prices)[-50:]
            
            ax.plot(x, y, color='#64C7FF', linewidth=2)
            ax.fill_between(x, y, alpha=0.3, color='#64C7FF')
            ax.set_ylim(0, 35)
        
        ax.set_title('Price Trend', color='#F0F0F0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#A0A0A0', labelsize=8)
        ax.spines['bottom'].set_color('#404060')
        ax.spines['left'].set_color('#404060')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, color='#404060')
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        return pygame.image.fromstring(raw_data, size, "RGB")
    
    @staticmethod
    def create_revenue_chart(tracker):
        """Create daily revenue chart"""
        fig = Figure(figsize=(4, 2), dpi=100, facecolor='#282828')
        ax = fig.add_subplot(111, facecolor='#1a1a2e')
        
        if len(tracker.daily_revenues) > 0:
            days = np.arange(len(tracker.daily_revenues))[-20:]
            revs = list(tracker.daily_revenues)[-20:]
            
            colors = ['#4CAF50' if i < len(revs) else '#FF6B6B' for i in range(len(revs))]
            ax.bar(days, revs, color='#76B74D', alpha=0.8, edgecolor='#A0D995')
            ax.set_ylim(0, max(revs) * 1.2 if revs else 100)
        
        ax.set_title('Daily Revenue', color='#F0F0F0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#A0A0A0', labelsize=8)
        ax.spines['bottom'].set_color('#404060')
        ax.spines['left'].set_color('#404060')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, axis='y', color='#404060')
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        return pygame.image.fromstring(raw_data, size, "RGB")
    
    @staticmethod
    def create_occupancy_chart(tracker):
        """Create occupancy trend chart"""
        fig = Figure(figsize=(4, 2), dpi=100, facecolor='#282828')
        ax = fig.add_subplot(111, facecolor='#1a1a2e')
        
        if len(tracker.occupancies) > 0:
            x = np.arange(len(tracker.occupancies))[-100:]
            y = [occ * 100 for occ in list(tracker.occupancies)[-100:]]
            
            ax.plot(x, y, color='#FF9800', linewidth=2)
            ax.axhline(y=60, color='#FF5722', linestyle='--', linewidth=1, alpha=0.7, label='Min 60%')
            ax.fill_between(x, y, alpha=0.2, color='#FF9800')
            ax.set_ylim(0, 120)
            ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#404060')
        
        ax.set_title('Occupancy %', color='#F0F0F0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#A0A0A0', labelsize=8)
        ax.spines['bottom'].set_color('#404060')
        ax.spines['left'].set_color('#404060')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, color='#404060')
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        return pygame.image.fromstring(raw_data, size, "RGB")


 
class Dashboard:
    """Real-time project dashboard"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🅿️ PARKING LOT PROJECT - REAL-TIME DASHBOARD")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Tracker
        self.tracker = MetricsTracker()
        
        # Chart cache (update every few frames)
        self.price_chart = None
        self.revenue_chart = None
        self.occupancy_chart = None
        self.chart_update_counter = 0
        self.chart_update_interval = 30
        
        # Fonts
        self.font_title = pygame.font.Font(None, 48)
        self.font_header = pygame.font.Font(None, 28)
        self.font_normal = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 14)
        
        # Demo mode - simulate data
        self.demo_time = 0
        self.demo_mode = True
    
    def update_charts(self):
        """Update chart images"""
        try:
            self.price_chart = ChartGenerator.create_price_chart(self.tracker)
            self.revenue_chart = ChartGenerator.create_revenue_chart(self.tracker)
            self.occupancy_chart = ChartGenerator.create_occupancy_chart(self.tracker)
        except:
            pass  # Charts may fail to render, continue anyway
    
    def simulate_data(self):
        """Simulate parking lot data for demo"""
        self.demo_time += 1
        
        # Simulate pricing events
        if self.demo_time % 5 == 0:
            hour = 6 + (self.demo_time % 720) // 60  # 6 AM to 6 PM
            base_price = 5.0 + (np.sin(self.demo_time * 0.01) + 1) * 10
            occupancy = 0.6 + 0.3 * np.sin(self.demo_time * 0.005)
            price = base_price + (occupancy ** 2) * 10
            
            self.tracker.add_price_event(price, occupancy)
            self.tracker.add_revenue_event(price)
            self.tracker.add_car_event()
            self.tracker.current_hour = hour % 24
        
        # Simulate day completion
        if self.demo_time % 144 == 0 and self.demo_time > 0:
            day_revenue = np.random.uniform(1000, 2500)
            day_cars = np.random.randint(50, 150)
            avg_price = day_revenue / day_cars
            avg_occupancy = np.random.uniform(0.65, 0.85)
            
            self.tracker.end_day(day_revenue, day_cars, avg_price, avg_occupancy)
            self.tracker.update_rl_status(
                max(0.01, 0.1 * (1 - self.demo_time/10000)),
                max(0.05, 0.2 * (1 - self.demo_time/10000)),
                self.tracker.days_simulated
            )
    
    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self, dt):
        """Update dashboard"""
        self.simulate_data()
        
        self.chart_update_counter += 1
        if self.chart_update_counter >= self.chart_update_interval:
            self.update_charts()
            self.chart_update_counter = 0
    
    def draw(self):
        """Draw entire dashboard"""
        self.screen.fill(COLORS['bg'])
        
        # Title
        self.draw_title()
        
        # Main sections
        self.draw_metrics_section()
        self.draw_charts_section()
        self.draw_rl_status_section()
        self.draw_project_summary()
        
        pygame.display.flip()
    
    def draw_title(self):
        """Draw dashboard title"""
        y = 15
        title = self.font_title.render("🅿️ PARKING LOT PRICING PROJECT - DASHBOARD", True, COLORS['accent'])
        self.screen.blit(title, (20, y))
        
        subtitle = self.font_small.render("Real-time Metrics & Analytics", True, COLORS['text_dim'])
        self.screen.blit(subtitle, (30, y + 45))
    
    def draw_metrics_section(self):
        """Draw key metrics cards"""
        x, y = 20, 80
        card_width = 280
        card_height = 180
        gap = 20
        
        # Card 1: Current Pricing
        self.draw_card(x, y, card_width, card_height, "💰 CURRENT PRICING")
        stats = self.tracker.get_statistics()
        metrics = [
            ("Price", f"${stats['avg_price']:.2f}"),
            ("Max", f"${stats['max_price']:.2f}"),
            ("Min", f"${stats['min_price']:.2f}"),
        ]
        self.draw_card_content(x, y, metrics)
        
        # Card 2: Occupancy
        self.draw_card(x + card_width + gap, y, card_width, card_height, "📊 OCCUPANCY")
        occupancy_pct = self.tracker.current_occupancy * 100
        metrics = [
            ("Current", f"{occupancy_pct:.1f}%"),
            ("Avg", f"{stats['avg_occupancy']*100:.1f}%"),
            ("Target", "60% min"),
        ]
        self.draw_card_content(x + card_width + gap, y, metrics)
        
        # Card 3: Revenue
        self.draw_card(x + (card_width + gap) * 2, y, card_width, card_height, "💵 REVENUE")
        metrics = [
            ("Total", f"${stats['total_revenue']:,.0f}"),
            ("Daily Avg", f"${stats['avg_daily_revenue']:,.0f}"),
            ("Days", f"{stats['days_simulated']}"),
        ]
        self.draw_card_content(x + (card_width + gap) * 2, y, metrics)
        
        # Card 4: Volume
        self.draw_card(x + (card_width + gap) * 3, y, card_width, card_height, "🚗 VOLUME")
        metrics = [
            ("Total Cars", f"{stats['total_cars']}"),
            ("Days", f"{stats['days_simulated']}"),
            ("Avg/Day", f"{stats['total_cars']/max(1, stats['days_simulated']):.0f}"),
        ]
        self.draw_card_content(x + (card_width + gap) * 3, y, metrics)
    
    def draw_card(self, x, y, width, height, title):
        """Draw metric card background"""
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x, y, width, height))
        pygame.draw.rect(self.screen, COLORS['card_border'], (x, y, width, height), 2)
        
        title_text = self.font_header.render(title, True, COLORS['accent'])
        self.screen.blit(title_text, (x + 15, y + 12))
    
    def draw_card_content(self, x, y, metrics):
        """Draw metrics inside card"""
        y_offset = y + 50
        for label, value in metrics:
            label_text = self.font_small.render(label, True, COLORS['text_dim'])
            value_text = self.font_normal.render(value, True, COLORS['success'])
            
            self.screen.blit(label_text, (x + 20, y_offset))
            self.screen.blit(value_text, (x + 20, y_offset + 25))
            y_offset += 55
    
    def draw_charts_section(self):
        """Draw chart section"""
        x, y = 20, 290
        chart_width = 400
        chart_height = 230
        gap = 15
        
        # Section title
        section_title = self.font_header.render("📈 TRENDS & ANALYTICS", True, COLORS['accent'])
        self.screen.blit(section_title, (x, y))
        y += 35
        
        # Chart backgrounds
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x, y, chart_width, chart_height))
        pygame.draw.rect(self.screen, COLORS['card_border'], (x, y, chart_width, chart_height), 1)
        
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x + chart_width + gap, y, chart_width, chart_height))
        pygame.draw.rect(self.screen, COLORS['card_border'], (x + chart_width + gap, y, chart_width, chart_height), 1)
        
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x + (chart_width + gap)*2, y, chart_width, chart_height))
        pygame.draw.rect(self.screen, COLORS['card_border'], (x + (chart_width + gap)*2, y, chart_width, chart_height), 1)
        
        # Draw charts
        if self.price_chart:
            self.screen.blit(self.price_chart, (x + 8, y + 8))
        
        if self.revenue_chart:
            self.screen.blit(self.revenue_chart, (x + chart_width + gap + 8, y + 8))
        
        if self.occupancy_chart:
            self.screen.blit(self.occupancy_chart, (x + (chart_width + gap)*2 + 8, y + 8))
    
    def draw_rl_status_section(self):
        """Draw RL model status"""
        x, y = 20, 560
        width = 900
        height = 150
        
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x, y, width, height))
        pygame.draw.rect(self.screen, COLORS['rl_accent'], (x, y, width, height), 2)
        
        title = self.font_header.render("🤖 RL MODEL STATUS", True, COLORS['rl_accent'])
        self.screen.blit(title, (x + 15, y + 12))
        
        y_offset = y + 50
        
        # Status info
        info_lines = [
            f"Training Days: {self.tracker.model_training_days}",
            f"Learning Rate: {self.tracker.learning_rate:.4f}",
            f"Exploration (ε): {self.tracker.epsilon:.3f}",
            f"Current Occupancy: {self.tracker.current_occupancy*100:.1f}%",
            f"Current Hour: {self.tracker.current_hour}:00",
            f"Weather: {self.tracker.current_weather}",
        ]
        
        for i, line in enumerate(info_lines):
            col = i % 3
            row = i // 3
            text = self.font_small.render(line, True, COLORS['text'])
            self.screen.blit(text, (x + 30 + col * 280, y_offset + row * 35))
    
    def draw_project_summary(self):
        """Draw project summary"""
        x, y = 20, 730
        width = 900
        height = 150
        
        pygame.draw.rect(self.screen, COLORS['card_bg'], (x, y, width, height))
        pygame.draw.rect(self.screen, COLORS['success'], (x, y, width, height), 2)
        
        title = self.font_header.render("📋 PROJECT SUMMARY", True, COLORS['success'])
        self.screen.blit(title, (x + 15, y + 12))
        
        y_offset = y + 50
        
        stats = self.tracker.get_statistics()
        
        summary_lines = [
            f" Simulation Days: {stats['days_simulated']}",
            f" Total Cars Parked: {stats['total_cars']}",
            f" Total Revenue: ${stats['total_revenue']:,.2f}",
            f" Avg Daily Revenue: ${stats['avg_daily_revenue']:,.2f}",
            f" Avg Price Point: ${stats['avg_price']:.2f}",
            f" Avg Occupancy: {stats['avg_occupancy']*100:.1f}%",
        ]
        
        for i, line in enumerate(summary_lines):
            col = i % 3
            row = i // 3
            text = self.font_small.render(line, True, COLORS['text'])
            self.screen.blit(text, (x + 30 + col * 280, y_offset + row * 35))
        
        # Footer
        footer = self.font_tiny.render("Press ESC to exit | Dashboard updates in real-time", True, COLORS['text_dim'])
        self.screen.blit(footer, (x + 15, y + 120))
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print(" PARKING LOT PROJECT - REAL-TIME DASHBOARD")
        print("="*80)
        print("\nDashboard Features:")
        print("  ✓ Real-time price and occupancy tracking")
        print("  ✓ Revenue trends and analytics")
        print("  ✓ RL model training progress")
        print("  ✓ Live metrics and statistics")
        print("  ✓ Project performance summary")
        print("\nDashboard running... (Update interval: 2 seconds)")
        print("="*80 + "\n")
        
        # Generate initial charts
        self.update_charts()
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()
        
        print("\n" + "="*80)
        print("FINAL PROJECT METRICS")
        print("="*80)
        stats = self.tracker.get_statistics()
        print(f"Days Simulated: {stats['days_simulated']}")
        print(f"Total Revenue: ${stats['total_revenue']:,.2f}")
        print(f"Total Cars: {stats['total_cars']}")
        print(f"Avg Price: ${stats['avg_price']:.2f}")
        print(f"Avg Occupancy: {stats['avg_occupancy']*100:.1f}%")
        print(f"RL Training Days: {self.tracker.model_training_days}")
        print("="*80 + "\n")


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
