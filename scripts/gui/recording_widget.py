"""
Recording widget for treadmill GUI
Includes record button and recording name input
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLineEdit, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor


class RecordingWidget(QWidget):
    """Widget for recording controls"""
    
    # Signals
    recording_started = pyqtSignal(str)  # Emits recording name
    recording_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Record button
        self.record_button = QPushButton()
        self.record_button.setFixedSize(60, 60)
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_recording)
        self.update_button_icon()
        
        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.record_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Recording name input
        name_layout = QVBoxLayout()
        name_layout.setSpacing(5)
        
        name_label = QLabel("Recording Name:")
        name_label.setAlignment(Qt.AlignCenter)
        name_layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        self.name_input.setText("test_00")
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setMaxLength(50)
        name_layout.addWidget(self.name_input)
        
        layout.addLayout(name_layout)
        
        self.setLayout(layout)
        
    def create_record_icon(self, is_recording):
        """Create a record button icon (red dot)"""
        pixmap = QPixmap(60, 60)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if is_recording:
            # Red circle when recording
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(10, 10, 40, 40)
        else:
            # Gray circle with red outline when not recording
            painter.setBrush(QColor(200, 200, 200))
            painter.setPen(QColor(255, 0, 0))
            painter.drawEllipse(10, 10, 40, 40)
        
        painter.end()
        return QIcon(pixmap)
        
    def update_button_icon(self):
        """Update the button icon based on recording state"""
        icon = self.create_record_icon(self.is_recording)
        self.record_button.setIcon(icon)
        self.record_button.setIconSize(self.record_button.size())
        
    def toggle_recording(self):
        """Toggle recording state"""
        self.is_recording = not self.is_recording
        self.update_button_icon()
        
        if self.is_recording:
            # Start recording
            recording_name = self.name_input.text().strip()
            if not recording_name:
                recording_name = "test_00"
                self.name_input.setText(recording_name)
            
            # Disable name input during recording
            self.name_input.setEnabled(False)
            
            self.recording_started.emit(recording_name)
        else:
            # Stop recording
            self.name_input.setEnabled(True)
            self.recording_stopped.emit()
            
    def get_recording_name(self):
        """Get the current recording name"""
        return self.name_input.text().strip()
        
    def reset(self):
        """Reset the widget to initial state"""
        if self.is_recording:
            self.is_recording = False
            self.update_button_icon()
            self.record_button.setChecked(False)
        self.name_input.setEnabled(True)
