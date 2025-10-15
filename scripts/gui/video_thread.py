"""
Video processing thread for Lab MoCap GUI
"""
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import os
from argparse import Namespace
from rtmlib import RTMDet, RTMPose
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR
from .test_gopro_stream import GoProCam


class VideoProcessingThread(QThread):
    """Thread for processing video frames with pose estimation"""
    
    # Signals
    frame_ready = pyqtSignal(object, list, list, list, dict)  # frame, track_data, keypoints, scores, angles
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.cameras = {}
        self.gopro_cam = None
        
        # Models
        self.detector = None
        self.tracker = None
        self.pose_estimator = None
        
        # Model paths
        self.rtmdet_model = os.path.join(MODEL_DIR, 'rtmdet-m-640.onnx')
        self.rtmpose_model = os.path.join(MODEL_DIR, 'rtmpose-m-256-192.onnx')
        
    def initialize_models(self):
        """Initialize detection, tracking, and pose estimation models"""
        try:
            # Initialize detector
            self.detector = RTMDet(
                onnx_model=self.rtmdet_model,
                model_input_size=(640, 640),
                backend='onnxruntime',
                device='cuda'
            )
            
            # Initialize tracker
            args = Namespace(
                track_thresh=0.5,
                match_thresh=0.8,
                track_buffer=30,
                frame_rate=30,
                mot20=False,
                min_hits=3
            )
            self.tracker = BYTETracker(args)
            
            # Initialize pose estimator
            self.pose_estimator = RTMPose(
                onnx_model=self.rtmpose_model,
                model_input_size=(192, 256),
                backend='onnxruntime',
                device='cuda'
            )
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize models: {str(e)}")
            return False
            
    def initialize_cameras(self):
        """Initialize camera captures based on configuration"""
        try:
            # Release any existing cameras
            self.release_cameras()
            
            self.cameras = {}
            
            if self.config.camera_mode == "single_ip":
                cam_id = self.config.selected_camera
                if cam_id not in self.config.camera_urls:
                    raise ValueError(f"Invalid camera selection: {cam_id}")
                self.cameras[cam_id] = cv2.VideoCapture(self.config.camera_urls[cam_id])
                
            elif self.config.camera_mode == "all_ip":
                for cam_id, url in self.config.camera_urls.items():
                    self.cameras[cam_id] = cv2.VideoCapture(url)
                    
            elif self.config.camera_mode == "single_gopro":
                # Initialize GoPro camera
                self.gopro_cam = GoProCam(
                    index=self.config.gopro_index,
                    size=self.config.gopro_size,
                    fps=self.config.gopro_fps,
                    fourcc=self.config.gopro_fourcc,
                    warmup_frames=self.config.gopro_warmup_frames
                )
                self.gopro_cam.open()
            
            # Enable OpenCV optimizations
            cv2.setUseOptimized(True)
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize cameras: {str(e)}")
            return False
            
    def release_cameras(self):
        """Release all camera resources"""
        for cap in self.cameras.values():
            if cap is not None:
                cap.release()
        self.cameras = {}
        
        # Release GoPro camera if active
        if self.gopro_cam is not None:
            self.gopro_cam.close()
            self.gopro_cam = None
        
    def capture_frame(self):
        """Capture frame(s) based on camera configuration"""
        if self.config.camera_mode == "single_gopro":
            # Capture from GoPro
            ret, frame = self.gopro_cam.read()
            if not ret:
                return None, None
            return frame, None  # No angle camera for single GoPro
        
        # Handle IP cameras
        frames = {}
        
        for cam_id, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                return None, None
            frames[cam_id] = frame
        
        if self.config.camera_mode == "single_ip":
            return frames[self.config.selected_camera], self.config.selected_camera
        elif self.config.camera_mode == "all_ip":
            # Stitch frames into 2x2 grid
            stitched = self.stitch_frames(frames[1], frames[2], frames[3], frames[4])
            return stitched, self.config.angle_computation_camera
            
    def stitch_frames(self, frame_1, frame_2, frame_3, frame_4):
        """Stitch 4 camera frames into a 2x2 grid"""
        size = (960, 540)
        f1 = cv2.resize(frame_1, size)
        f2 = cv2.resize(frame_2, size)
        f3 = cv2.resize(frame_3, size)
        f4 = cv2.resize(frame_4, size)
        
        top = np.hstack((f1, f2))
        bottom = np.hstack((f3, f4))
        combined = np.vstack((top, bottom))
        return combined
        
    def run(self):
        """Main processing loop"""
        # Initialize models
        if not self.initialize_models():
            return
            
        # Initialize cameras
        if not self.initialize_cameras():
            return
            
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                frame, angle_camera = self.capture_frame()
                if frame is None:
                    continue
                
                height, width = frame.shape[:2]
                
                # Step 1: Detection
                det_bboxes_scores, _ = self.detector(frame)
                det_bboxes, det_scores = det_bboxes_scores
                
                # Step 2: Format for ByteTrack
                if len(det_bboxes) > 0:
                    dets_for_tracker = np.array([[*box, score, 0] 
                                                for box, score in zip(det_bboxes, det_scores)])
                else:
                    dets_for_tracker = np.empty((0, 6))
                
                # Step 3: Tracking
                tracks = self.tracker.update(dets_for_tracker, [height, width], (height, width))
                
                # Step 4: Prepare data for pose estimation
                track_data = []
                tracked_bboxes = []
                
                for track in tracks:
                    if not track.is_activated:
                        continue
                    
                    x1, y1, w, h = track.tlwh
                    x2, y2 = x1 + w, y1 + h
                    track_id = int(track.track_id)
                    score = track.score if hasattr(track, "score") else 0.0
                    
                    track_data.append({
                        'id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'score': score
                    })
                    tracked_bboxes.append([x1, y1, x2, y2])
                
                # Step 5: Pose estimation
                keypoints_list = []
                scores_list = []
                
                if len(tracked_bboxes) > 0:
                    keypoints_array, scores_array, _ = self.pose_estimator(frame, tracked_bboxes)
                    # Convert numpy arrays to lists for PyQt5 signal
                    keypoints_list = [kp.tolist() if isinstance(kp, np.ndarray) else kp for kp in keypoints_array]
                    scores_list = [sc.tolist() if isinstance(sc, np.ndarray) else sc for sc in scores_array]
                
                # Step 6: Calculate angles for tracked person
                # In "all" mode, we need to get keypoints from the specific camera
                # For now, we'll use the keypoints from the stitched frame
                # (This is a simplification - in production you'd extract from specific camera)
                angles = {}
                
                # Emit processed frame data
                self.frame_ready.emit(frame, track_data, keypoints_list, scores_list, angles)
                
        except Exception as e:
            self.error_occurred.emit(f"Error in processing loop: {str(e)}")
        finally:
            self.release_cameras()
            
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()
