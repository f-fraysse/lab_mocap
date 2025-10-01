"""
Main window for Lab MoCap GUI
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QMenuBar, QAction, QMessageBox,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

from .config import AppConfig
from .input_dialog import InputDialog
from .display_dialog import DisplayDialog
from .video_thread import VideoProcessingThread
from .angle_graph_widget import AngleGraphWidget
from .angle_calculator import calculate_joint_angle


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.video_thread = None
        self.current_track_ids = set()
        
        self.setWindowTitle("Lab MoCap - Human Pose Estimation")
        self.setGeometry(100, 100, 1600, 900)
        
        self.init_ui()
        self.init_menu()
        
        # Start video processing
        self.start_video_processing()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left sidebar for angle graphs
        left_sidebar = self.create_left_sidebar()
        main_layout.addWidget(left_sidebar)
        
        # Main display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        main_layout.addWidget(self.video_label, stretch=1)
        
        central_widget.setLayout(main_layout)
        
    def create_left_sidebar(self):
        """Create the left sidebar with angle graphs"""
        sidebar = QWidget()
        sidebar.setMaximumWidth(400)
        sidebar.setMinimumWidth(350)
        layout = QVBoxLayout()
        
        # Track ID selection
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Track ID:"))
        self.track_id_combo = QComboBox()
        self.track_id_combo.addItem("None")
        self.track_id_combo.currentTextChanged.connect(self.on_track_id_changed)
        # Populate dropdown when clicked
        self.track_id_combo.showPopup = self.populate_track_ids
        id_layout.addWidget(self.track_id_combo)
        layout.addLayout(id_layout)
        
        # Graph 1
        layout.addWidget(QLabel("Joint 1:"))
        self.joint1_combo = QComboBox()
        self.joint1_combo.addItems(["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Elbow", "Right Elbow"])
        self.joint1_combo.setCurrentText("Left Hip")
        layout.addWidget(self.joint1_combo)
        
        self.graph1 = AngleGraphWidget()
        layout.addWidget(self.graph1)
        
        # Graph 2
        layout.addWidget(QLabel("Joint 2:"))
        self.joint2_combo = QComboBox()
        self.joint2_combo.addItems(["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Elbow", "Right Elbow"])
        self.joint2_combo.setCurrentText("Left Knee")
        layout.addWidget(self.joint2_combo)
        
        self.graph2 = AngleGraphWidget()
        layout.addWidget(self.graph2)
        
        # Graph 3
        layout.addWidget(QLabel("Joint 3:"))
        self.joint3_combo = QComboBox()
        self.joint3_combo.addItems(["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Elbow", "Right Elbow"])
        self.joint3_combo.setCurrentText("Left Elbow")
        layout.addWidget(self.joint3_combo)
        
        self.graph3 = AngleGraphWidget()
        layout.addWidget(self.graph3)
        
        layout.addStretch()
        sidebar.setLayout(layout)
        return sidebar
        
    def init_menu(self):
        """Initialize the menu bar"""
        menubar = self.menuBar()
        
        # Input menu
        input_menu = menubar.addMenu("Input")
        input_action = QAction("Configure Input...", self)
        input_action.triggered.connect(self.show_input_dialog)
        input_menu.addAction(input_action)
        
        # Display menu
        display_menu = menubar.addMenu("Display")
        display_action = QAction("Display Options...", self)
        display_action.triggered.connect(self.show_display_dialog)
        display_menu.addAction(display_action)
        
    def show_input_dialog(self):
        """Show input configuration dialog"""
        dialog = InputDialog(self.config, self)
        if dialog.exec_():
            # Configuration changed, restart video processing
            self.restart_video_processing()
            
    def show_display_dialog(self):
        """Show display options dialog"""
        dialog = DisplayDialog(self.config, self)
        dialog.exec_()
        # Display options are applied immediately, no restart needed
        
    def populate_track_ids(self):
        """Populate track ID dropdown with current IDs"""
        # Store current selection
        current_text = self.track_id_combo.currentText()
        
        # Clear and repopulate
        self.track_id_combo.clear()
        self.track_id_combo.addItem("None")
        
        for track_id in sorted(self.current_track_ids):
            self.track_id_combo.addItem(str(track_id))
        
        # Restore selection if still valid
        index = self.track_id_combo.findText(current_text)
        if index >= 0:
            self.track_id_combo.setCurrentIndex(index)
        
        # Show the popup
        QComboBox.showPopup(self.track_id_combo)
        
    def on_track_id_changed(self, text):
        """Handle track ID selection change"""
        if text == "None":
            self.config.selected_track_id = None
            # Clear graphs
            self.graph1.clear_data()
            self.graph2.clear_data()
            self.graph3.clear_data()
        else:
            try:
                self.config.selected_track_id = int(text)
                # Reset graph time references
                self.graph1.reset_time()
                self.graph2.reset_time()
                self.graph3.reset_time()
            except ValueError:
                self.config.selected_track_id = None
                
    def start_video_processing(self):
        """Start the video processing thread"""
        self.video_thread = VideoProcessingThread(self.config)
        self.video_thread.frame_ready.connect(self.update_display)
        self.video_thread.error_occurred.connect(self.show_error)
        self.video_thread.start()
        
    def restart_video_processing(self):
        """Restart video processing with new configuration"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        
        # Clear current track IDs
        self.current_track_ids.clear()
        self.config.selected_track_id = None
        self.track_id_combo.setCurrentText("None")
        
        # Clear graphs
        self.graph1.clear_data()
        self.graph2.clear_data()
        self.graph3.clear_data()
        
        self.start_video_processing()
        
    def update_display(self, frame, track_data, keypoints_list, scores_list, angles):
        """Update display with new frame data"""
        # Draw overlays on frame
        display_frame = self.draw_overlays(frame.copy(), track_data, keypoints_list, scores_list)
        
        # Update track IDs
        self.current_track_ids = {track['id'] for track in track_data}
        
        # Update angle graphs if a track is selected
        if self.config.selected_track_id is not None:
            self.update_angle_graphs(track_data, keypoints_list, scores_list)
        
        # Convert frame to QPixmap and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def draw_overlays(self, frame, track_data, keypoints_list, scores_list):
        """Draw overlays on frame based on display configuration"""
        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for i, track in enumerate(track_data):
            track_id = track['id']
            bbox = track['bbox']
            score = track['score']
            
            # Draw bounding box
            if self.config.show_bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw track ID
            if self.config.show_track_ids:
                x1, y1 = int(bbox[0]), int(bbox[1])
                label = f"ID: {track_id}"
                if score is not None:
                    label += f" | {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw keypoints and skeleton
            if i < len(keypoints_list):
                # Convert lists back to numpy arrays for processing
                keypoints = np.array(keypoints_list[i])
                kpt_scores = np.array(scores_list[i])
                
                # Draw skeleton connections
                if self.config.show_skeleton:
                    connections = self.config.get_active_connections()
                    for conn in connections:
                        idx1, idx2 = conn['indices']
                        if kpt_scores[idx1] > 0.3 and kpt_scores[idx2] > 0.3:
                            pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                            pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                            # Convert BGR to RGB for display
                            color = (conn['color'][2], conn['color'][1], conn['color'][0])
                            cv2.line(frame, pt1, pt2, color, conn['thickness'])
                
                # Draw keypoints
                if self.config.show_keypoints:
                    for j, (kpt, kpt_score) in enumerate(zip(keypoints, kpt_scores)):
                        if kpt_score > 0.3:
                            pt = (int(kpt[0]), int(kpt[1]))
                            # Use green for left side, orange for right side
                            color = (0, 255, 0) if j % 2 == 1 else (255, 128, 0)
                            cv2.circle(frame, pt, self.config.keypoint_size, color, -1)
        
        return frame
        
    def update_angle_graphs(self, track_data, keypoints_list, scores_list):
        """Update angle graphs for selected track"""
        # Find the selected track
        track_idx = None
        for i, track in enumerate(track_data):
            if track['id'] == self.config.selected_track_id:
                track_idx = i
                break
        
        if track_idx is None or track_idx >= len(keypoints_list):
            # Track not found or no keypoints, clear graphs
            return
        
        keypoints = keypoints_list[track_idx]
        scores = scores_list[track_idx]
        
        # Calculate angles for each graph
        # Parse joint selection (e.g., "Left Hip" -> side='left', joint='hip')
        selection1 = self.joint1_combo.currentText()
        parts1 = selection1.split()
        side1 = parts1[0].lower()  # 'left' or 'right'
        joint1 = parts1[1].lower()  # 'hip', 'knee', or 'elbow'
        angle1 = calculate_joint_angle(joint1, keypoints, scores, side=side1)
        if angle1 is not None:
            self.graph1.add_data_point(angle1)
        
        selection2 = self.joint2_combo.currentText()
        parts2 = selection2.split()
        side2 = parts2[0].lower()
        joint2 = parts2[1].lower()
        angle2 = calculate_joint_angle(joint2, keypoints, scores, side=side2)
        if angle2 is not None:
            self.graph2.add_data_point(angle2)
        
        selection3 = self.joint3_combo.currentText()
        parts3 = selection3.split()
        side3 = parts3[0].lower()
        joint3 = parts3[1].lower()
        angle3 = calculate_joint_angle(joint3, keypoints, scores, side=side3)
        if angle3 is not None:
            self.graph3.add_data_point(angle3)
            
    def show_error(self, error_message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", error_message)
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()
