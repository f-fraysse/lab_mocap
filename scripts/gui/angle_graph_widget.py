"""
Angle graph widget using matplotlib
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque
import time


class AngleGraphWidget(QWidget):
    """Widget for displaying joint angle time history"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage (timestamp, angle) tuples
        self.angle_history = deque(maxlen=1000)
        self.history_duration = 5.0  # seconds
        self.start_time = time.time()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(4, 3), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configure plot
        self.ax.set_xlim(0, self.history_duration)
        self.ax.set_ylim(-5, 180)
        self.ax.set_xlabel('Time (s)', fontsize=9)
        self.ax.set_ylabel('Angle (deg)', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(labelsize=8)
        
        # Initialize empty plot line
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Tight layout
        self.figure.tight_layout()
        
    def add_data_point(self, angle):
        """Add a new angle data point"""
        current_time = time.time() - self.start_time
        self.angle_history.append((current_time, angle))
        self.update_plot()
        
    def update_plot(self):
        """Update the plot with current data"""
        if len(self.angle_history) == 0:
            self.line.set_data([], [])
            self.canvas.draw_idle()
            return
        
        # Get current time
        current_time = time.time() - self.start_time
        
        # Filter to last 5 seconds
        time_start = current_time - self.history_duration
        recent_data = [(t, angle) for t, angle in self.angle_history if t >= time_start]
        
        if len(recent_data) > 0:
            # Extract times and angles
            times = [t - time_start for t, _ in recent_data]  # Relative to window start
            angles = [angle for _, angle in recent_data]
            
            # Update line data
            self.line.set_data(times, angles)
            
            # Update x-axis to show rolling window
            self.ax.set_xlim(0, self.history_duration)
        else:
            self.line.set_data([], [])
        
        # Redraw canvas
        self.canvas.draw_idle()
        
    def clear_data(self):
        """Clear all data from the graph"""
        self.angle_history.clear()
        self.line.set_data([], [])
        self.canvas.draw_idle()
        
    def reset_time(self):
        """Reset the time reference"""
        self.start_time = time.time()
        self.angle_history.clear()
        self.line.set_data([], [])
        self.canvas.draw_idle()
