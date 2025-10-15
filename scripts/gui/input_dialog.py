"""
Input configuration dialog
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QRadioButton, QComboBox, QLabel, QPushButton,
                             QButtonGroup)
from PyQt5.QtCore import Qt


class InputDialog(QDialog):
    """Dialog for configuring input stream settings"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Input Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # Store original values for cancel
        self.original_camera_mode = config.camera_mode
        self.original_selected_camera = config.selected_camera
        self.original_angle_camera = config.angle_computation_camera
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Camera Mode Group
        mode_group = QGroupBox("Camera Mode")
        mode_layout = QVBoxLayout()
        
        self.single_ip_radio = QRadioButton("Single IP Camera")
        self.all_ip_radio = QRadioButton("All IP Cameras")
        self.single_gopro_radio = QRadioButton("Single GoPro Camera")
        
        # Button group for mutual exclusivity
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.single_ip_radio)
        self.mode_button_group.addButton(self.all_ip_radio)
        self.mode_button_group.addButton(self.single_gopro_radio)
        
        mode_layout.addWidget(self.single_ip_radio)
        mode_layout.addWidget(self.all_ip_radio)
        mode_layout.addWidget(self.single_gopro_radio)
        mode_group.setLayout(mode_layout)
        
        # Set current mode
        if self.config.camera_mode == "single_ip":
            self.single_ip_radio.setChecked(True)
        elif self.config.camera_mode == "all_ip":
            self.all_ip_radio.setChecked(True)
        else:  # single_gopro
            self.single_gopro_radio.setChecked(True)
        
        layout.addWidget(mode_group)
        
        # Single Camera Selection Group
        self.single_group = QGroupBox("Single Camera Selection")
        single_layout = QHBoxLayout()
        single_layout.addWidget(QLabel("Camera:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["1", "2", "3", "4"])
        self.camera_combo.setCurrentIndex(self.config.selected_camera - 1)
        
        single_layout.addWidget(self.camera_combo)
        single_layout.addStretch()
        self.single_group.setLayout(single_layout)
        
        layout.addWidget(self.single_group)
        
        # All Cameras - Angle Computation Selection Group
        self.all_group = QGroupBox("Angle Computation Camera (All Cameras Mode)")
        all_layout = QHBoxLayout()
        all_layout.addWidget(QLabel("Camera:"))
        
        self.angle_camera_combo = QComboBox()
        self.angle_camera_combo.addItems(["1", "2", "3", "4"])
        self.angle_camera_combo.setCurrentIndex(self.config.angle_computation_camera - 1)
        
        all_layout.addWidget(self.angle_camera_combo)
        all_layout.addStretch()
        self.all_group.setLayout(all_layout)
        
        layout.addWidget(self.all_group)
        
        # Update group visibility based on mode
        self.update_group_visibility()
        
        # Connect mode change signals
        self.single_ip_radio.toggled.connect(self.update_group_visibility)
        self.all_ip_radio.toggled.connect(self.update_group_visibility)
        self.single_gopro_radio.toggled.connect(self.update_group_visibility)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def update_group_visibility(self):
        """Update visibility of camera selection groups based on mode"""
        if self.single_ip_radio.isChecked():
            self.single_group.setEnabled(True)
            self.all_group.setEnabled(False)
        elif self.all_ip_radio.isChecked():
            self.single_group.setEnabled(False)
            self.all_group.setEnabled(True)
        else:  # single_gopro
            self.single_group.setEnabled(False)
            self.all_group.setEnabled(False)
            
    def apply_changes(self):
        """Apply configuration changes"""
        # Update camera mode
        if self.single_ip_radio.isChecked():
            self.config.camera_mode = "single_ip"
            self.config.selected_camera = int(self.camera_combo.currentText())
        elif self.all_ip_radio.isChecked():
            self.config.camera_mode = "all_ip"
            self.config.angle_computation_camera = int(self.angle_camera_combo.currentText())
        else:  # single_gopro
            self.config.camera_mode = "single_gopro"
        
        self.accept()
        
    def reject(self):
        """Cancel changes and restore original values"""
        self.config.camera_mode = self.original_camera_mode
        self.config.selected_camera = self.original_selected_camera
        self.config.angle_computation_camera = self.original_angle_camera
        super().reject()
