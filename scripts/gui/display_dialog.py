"""
Display options configuration dialog
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QCheckBox, QLabel, QPushButton, QSlider,
                             QColorDialog, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import copy


class DisplayDialog(QDialog):
    """Dialog for configuring display options"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Display Options")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        # Store original values for cancel
        self.original_show_bboxes = config.show_bboxes
        self.original_show_track_ids = config.show_track_ids
        self.original_show_keypoints = config.show_keypoints
        self.original_keypoint_size = config.keypoint_size
        self.original_show_skeleton = config.show_skeleton
        self.original_skeleton_groups = copy.deepcopy(config.skeleton_groups)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Basic Display Options Group
        basic_group = QGroupBox("Basic Display Options")
        basic_layout = QVBoxLayout()
        
        self.bbox_check = QCheckBox("Show Bounding Boxes")
        self.bbox_check.setChecked(self.config.show_bboxes)
        basic_layout.addWidget(self.bbox_check)
        
        self.id_check = QCheckBox("Show Track IDs")
        self.id_check.setChecked(self.config.show_track_ids)
        basic_layout.addWidget(self.id_check)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Keypoints Group
        keypoint_group = QGroupBox("Keypoints")
        keypoint_layout = QVBoxLayout()
        
        self.keypoint_check = QCheckBox("Show Keypoints")
        self.keypoint_check.setChecked(self.config.show_keypoints)
        keypoint_layout.addWidget(self.keypoint_check)
        
        # Keypoint size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.keypoint_slider = QSlider(Qt.Horizontal)
        self.keypoint_slider.setMinimum(1)
        self.keypoint_slider.setMaximum(10)
        self.keypoint_slider.setValue(self.config.keypoint_size)
        self.keypoint_slider.setTickPosition(QSlider.TicksBelow)
        self.keypoint_slider.setTickInterval(1)
        size_layout.addWidget(self.keypoint_slider)
        self.keypoint_size_label = QLabel(str(self.config.keypoint_size))
        size_layout.addWidget(self.keypoint_size_label)
        self.keypoint_slider.valueChanged.connect(
            lambda v: self.keypoint_size_label.setText(str(v)))
        keypoint_layout.addLayout(size_layout)
        
        keypoint_group.setLayout(keypoint_layout)
        layout.addWidget(keypoint_group)
        
        # Skeleton Connections Group
        skeleton_group = QGroupBox("Skeleton Connections")
        skeleton_layout = QVBoxLayout()
        
        self.skeleton_check = QCheckBox("Show All Connections")
        self.skeleton_check.setChecked(self.config.show_skeleton)
        skeleton_layout.addWidget(self.skeleton_check)
        
        # Individual group controls
        self.group_widgets = {}
        
        group_names = {
            'left_leg': 'Left Leg',
            'right_leg': 'Right Leg',
            'left_arm': 'Left Arm',
            'right_arm': 'Right Arm',
            'torso': 'Torso',
            'head': 'Head'
        }
        
        for group_key, group_label in group_names.items():
            group_config = self.config.skeleton_groups[group_key]
            
            group_widget_layout = QHBoxLayout()
            
            # Checkbox
            check = QCheckBox(group_label)
            check.setChecked(group_config['enabled'])
            group_widget_layout.addWidget(check)
            
            # Color button
            color_btn = QPushButton("Color")
            color_btn.setMaximumWidth(80)
            # Store BGR color
            bgr_color = group_config['color']
            # Convert BGR to RGB for Qt
            rgb_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])
            color_btn.setStyleSheet(f"background-color: {rgb_color.name()};")
            color_btn.clicked.connect(lambda checked, btn=color_btn, key=group_key: 
                                     self.choose_color(btn, key))
            group_widget_layout.addWidget(color_btn)
            
            # Thickness slider
            group_widget_layout.addWidget(QLabel("Thickness:"))
            thickness_slider = QSlider(Qt.Horizontal)
            thickness_slider.setMinimum(1)
            thickness_slider.setMaximum(10)
            thickness_slider.setValue(group_config['thickness'])
            thickness_slider.setMaximumWidth(100)
            group_widget_layout.addWidget(thickness_slider)
            
            thickness_label = QLabel(str(group_config['thickness']))
            thickness_slider.valueChanged.connect(
                lambda v, lbl=thickness_label: lbl.setText(str(v)))
            group_widget_layout.addWidget(thickness_label)
            
            group_widget_layout.addStretch()
            
            skeleton_layout.addLayout(group_widget_layout)
            
            # Store widgets for later access
            self.group_widgets[group_key] = {
                'check': check,
                'color_btn': color_btn,
                'thickness_slider': thickness_slider,
                'thickness_label': thickness_label
            }
        
        skeleton_group.setLayout(skeleton_layout)
        layout.addWidget(skeleton_group)
        
        layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_defaults)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
    def choose_color(self, button, group_key):
        """Open color picker dialog"""
        # Get current color (BGR)
        current_bgr = self.config.skeleton_groups[group_key]['color']
        current_rgb = QColor(current_bgr[2], current_bgr[1], current_bgr[0])
        
        # Open color dialog
        color = QColorDialog.getColor(current_rgb, self, "Choose Color")
        
        if color.isValid():
            # Update button color
            button.setStyleSheet(f"background-color: {color.name()};")
            # Store BGR color
            self.config.skeleton_groups[group_key]['color'] = (
                color.blue(), color.green(), color.red()
            )
            
    def reset_defaults(self):
        """Reset all display options to defaults"""
        self.config.reset_display_defaults()
        
        # Update UI to reflect defaults
        self.bbox_check.setChecked(self.config.show_bboxes)
        self.id_check.setChecked(self.config.show_track_ids)
        self.keypoint_check.setChecked(self.config.show_keypoints)
        self.keypoint_slider.setValue(self.config.keypoint_size)
        self.skeleton_check.setChecked(self.config.show_skeleton)
        
        # Update group widgets
        for group_key, widgets in self.group_widgets.items():
            group_config = self.config.skeleton_groups[group_key]
            widgets['check'].setChecked(group_config['enabled'])
            
            # Update color button
            bgr_color = group_config['color']
            rgb_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])
            widgets['color_btn'].setStyleSheet(f"background-color: {rgb_color.name()};")
            
            # Update thickness
            widgets['thickness_slider'].setValue(group_config['thickness'])
            
    def apply_changes(self):
        """Apply configuration changes"""
        # Update basic options
        self.config.show_bboxes = self.bbox_check.isChecked()
        self.config.show_track_ids = self.id_check.isChecked()
        self.config.show_keypoints = self.keypoint_check.isChecked()
        self.config.keypoint_size = self.keypoint_slider.value()
        self.config.show_skeleton = self.skeleton_check.isChecked()
        
        # Update skeleton groups
        for group_key, widgets in self.group_widgets.items():
            self.config.skeleton_groups[group_key]['enabled'] = widgets['check'].isChecked()
            self.config.skeleton_groups[group_key]['thickness'] = widgets['thickness_slider'].value()
            # Color is already updated in choose_color method
        
        self.accept()
        
    def reject(self):
        """Cancel changes and restore original values"""
        self.config.show_bboxes = self.original_show_bboxes
        self.config.show_track_ids = self.original_show_track_ids
        self.config.show_keypoints = self.original_show_keypoints
        self.config.keypoint_size = self.original_keypoint_size
        self.config.show_skeleton = self.original_show_skeleton
        self.config.skeleton_groups = self.original_skeleton_groups
        super().reject()
