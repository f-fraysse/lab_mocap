"""
Main window for Treadmill Analysis GUI
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
import csv
from threading import Thread

from .config import AppConfig
from .treadmill_video_thread import TreadmillVideoThread
from .treadmill_angle_graph import TreadmillAngleGraph
from .recording_widget import RecordingWidget
from .angle_calculator import calculate_hip_angle, calculate_knee_angle, LEFT_ANKLE_IDX
from .gait_analysis import analyze_recording


class TreadmillMainWindow(QMainWindow):
    """Main application window for treadmill analysis"""
    
    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.video_thread = None
        
        # Recording state
        self.is_recording = False
        self.recording_name = ""
        self.recording_data = {
            'hip_angles': [],
            'knee_angles': [],
            'ankle_coords': [],
            'frames': []
        }
        self.frame_count = 0
        
        self.setWindowTitle("Lab MoCap - Treadmill Analysis")
        self.setGeometry(100, 100, 1600, 900)
        
        self.init_ui()
        
        # Start video processing
        self.start_video_processing()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left sidebar for angle graphs and recording controls
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
        """Create the left sidebar with angle graphs and recording controls"""
        sidebar = QWidget()
        sidebar.setMaximumWidth(400)
        sidebar.setMinimumWidth(350)
        layout = QVBoxLayout()
        
        # Hip angle graph (fixed Y-axis: -30 to +100)
        self.hip_graph = TreadmillAngleGraph("Left Hip", -30, 100)
        layout.addWidget(self.hip_graph)
        
        # Knee angle graph (fixed Y-axis: -10 to +160)
        self.knee_graph = TreadmillAngleGraph("Left Knee", -10, 160)
        layout.addWidget(self.knee_graph)
        
        # Recording widget
        self.recording_widget = RecordingWidget()
        self.recording_widget.recording_started.connect(self.on_recording_started)
        self.recording_widget.recording_stopped.connect(self.on_recording_stopped)
        layout.addWidget(self.recording_widget)
        
        layout.addStretch()
        sidebar.setLayout(layout)
        return sidebar
        
    def start_video_processing(self):
        """Start the video processing thread"""
        self.video_thread = TreadmillVideoThread(self.config)
        self.video_thread.frame_ready.connect(self.update_display)
        self.video_thread.error_occurred.connect(self.show_error)
        self.video_thread.start()
        
    def update_display(self, frame, track_data, keypoints_list, scores_list):
        """Update display with new frame data"""
        # Draw overlays on frame
        display_frame = self.draw_overlays(frame.copy(), track_data, keypoints_list, scores_list)
        
        # Find lowest track ID (automatic selection)
        if len(track_data) > 0:
            # Sort by track ID and get the lowest
            sorted_tracks = sorted(track_data, key=lambda x: x['id'])
            selected_track_id = sorted_tracks[0]['id']
            
            # Find index of selected track
            track_idx = None
            for i, track in enumerate(track_data):
                if track['id'] == selected_track_id:
                    track_idx = i
                    break
            
            # Update angle graphs if track found
            if track_idx is not None and track_idx < len(keypoints_list):
                keypoints = np.array(keypoints_list[track_idx])
                scores = np.array(scores_list[track_idx])
                
                # Calculate hip angle
                hip_angle = calculate_hip_angle(keypoints, scores, side='left')
                if hip_angle is not None:
                    self.hip_graph.add_data_point(hip_angle)
                    
                    # Record if recording is active
                    if self.is_recording:
                        self.recording_data['hip_angles'].append(hip_angle)
                
                # Calculate knee angle
                knee_angle = calculate_knee_angle(keypoints, scores, side='left')
                if knee_angle is not None:
                    self.knee_graph.add_data_point(knee_angle)
                    
                    # Record if recording is active
                    if self.is_recording:
                        self.recording_data['knee_angles'].append(knee_angle)
                
                # Get ankle coordinates for recording
                if self.is_recording:
                    ankle_coord = keypoints[LEFT_ANKLE_IDX]
                    self.recording_data['ankle_coords'].append((ankle_coord[0], ankle_coord[1]))
        
        # Save frame if recording
        if self.is_recording:
            self.recording_data['frames'].append(display_frame.copy())
            self.frame_count += 1
        
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
        
    def on_recording_started(self, recording_name):
        """Handle recording start"""
        self.is_recording = True
        self.recording_name = recording_name
        self.frame_count = 0
        self.recording_data = {
            'hip_angles': [],
            'knee_angles': [],
            'ankle_coords': [],
            'frames': []
        }
        print(f"Recording started: {recording_name}")
        
    def on_recording_stopped(self):
        """Handle recording stop and trigger analysis"""
        self.is_recording = False
        print(f"Recording stopped: {self.recording_name}")
        print(f"Recorded {self.frame_count} frames")
        
        # Run analysis in separate thread to avoid blocking GUI
        analysis_thread = Thread(target=self.analyze_and_save_recording)
        analysis_thread.start()
        
    def analyze_and_save_recording(self):
        """Analyze recording and save all outputs"""
        try:
            # Create output directory
            output_dir = os.path.join('output', self.recording_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV file
            self.save_csv(output_dir)
            
            # Save video
            self.save_video(output_dir)
            
            # Perform gait analysis if we have enough data
            if len(self.recording_data['hip_angles']) > 30:  # At least 1 second of data
                print("Performing gait analysis...")
                results = analyze_recording(
                    self.recording_data['hip_angles'],
                    self.recording_data['knee_angles'],
                    self.recording_data['ankle_coords'],
                    output_dir,
                    self.recording_name,
                    fps=30,
                    frames=self.recording_data['frames']
                )
                print(f"Analysis complete: {results['num_strides']} strides detected")
            else:
                print("Not enough data for gait analysis")
            
            print(f"All outputs saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            self.show_error(f"Error during analysis: {str(e)}")
            
    def save_csv(self, output_dir):
        """Save recording data to CSV file"""
        csv_path = os.path.join(output_dir, f'{self.recording_name}.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['frame', 'time_s', 'hip_angle', 'knee_angle', 'ankle_x', 'ankle_y'])
            
            # Write data
            num_frames = len(self.recording_data['hip_angles'])
            for i in range(num_frames):
                frame_num = i
                time_s = i / 30.0  # 30 fps
                hip_angle = self.recording_data['hip_angles'][i] if i < len(self.recording_data['hip_angles']) else ''
                knee_angle = self.recording_data['knee_angles'][i] if i < len(self.recording_data['knee_angles']) else ''
                ankle_x, ankle_y = self.recording_data['ankle_coords'][i] if i < len(self.recording_data['ankle_coords']) else ('', '')
                
                writer.writerow([frame_num, f'{time_s:.3f}', hip_angle, knee_angle, ankle_x, ankle_y])
        
        print(f"CSV saved: {csv_path}")
        
    def save_video(self, output_dir):
        """Save recorded video frames"""
        video_path = os.path.join(output_dir, f'{self.recording_name}.mp4')
        
        if len(self.recording_data['frames']) == 0:
            print("No frames to save")
            return
        
        # Get frame dimensions
        height, width = self.recording_data['frames'][0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        
        # Write frames (convert RGB back to BGR for video)
        for frame in self.recording_data['frames']:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
        
        video_writer.release()
        print(f"Video saved: {video_path}")
        
    def show_error(self, error_message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", error_message)
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()
