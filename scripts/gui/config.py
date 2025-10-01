"""
Configuration management for Lab MoCap GUI
"""

class AppConfig:
    """Application configuration state"""
    
    def __init__(self):
        # Camera configuration
        self.camera_mode = "single"  # "single" or "all"
        self.selected_camera = 4  # 1-4
        self.angle_computation_camera = 4  # 1-4 (used in "all" mode)
        
        # Camera URLs (hardcoded RTSP addresses)
        self.camera_urls = {
            1: "rtsp://ubnt:ubnt@192.168.5.41:554/s0",
            2: "rtsp://ubnt:ubnt@192.168.5.45:554/s0", 
            3: "rtsp://ubnt:ubnt@192.168.5.42:554/s0",
            4: "rtsp://ubnt:ubnt@192.168.5.48:554/s0"
        }
        
        # Display options
        self.show_bboxes = True
        self.show_track_ids = True
        self.show_keypoints = True
        self.keypoint_size = 3
        self.show_skeleton = True
        
        # Skeleton group settings (group_name: {enabled, color, thickness})
        # Colors in BGR format for OpenCV
        self.skeleton_groups = {
            'left_leg': {'enabled': True, 'color': (0, 255, 0), 'thickness': 2},  # Green
            'right_leg': {'enabled': True, 'color': (0, 128, 255), 'thickness': 2},  # Orange
            'left_arm': {'enabled': True, 'color': (0, 255, 0), 'thickness': 2},  # Green
            'right_arm': {'enabled': True, 'color': (0, 128, 255), 'thickness': 2},  # Orange
            'torso': {'enabled': True, 'color': (255, 0, 255), 'thickness': 2},  # Magenta
            'head': {'enabled': True, 'color': (255, 153, 51), 'thickness': 2}  # Light blue
        }
        
        # Skeleton group connections (COCO17 keypoint indices)
        self.skeleton_connections = {
            'left_leg': [(11, 13), (13, 15)],  # hip-knee-ankle
            'right_leg': [(12, 14), (14, 16)],
            'left_arm': [(5, 7), (7, 9)],  # shoulder-elbow-wrist
            'right_arm': [(6, 8), (8, 10)],
            'torso': [(5, 6), (5, 11), (6, 12), (11, 12)],  # shoulders and hips
            'head': [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]  # nose-eyes-ears-shoulders
        }
        
        # Angle tracking
        self.selected_track_id = None  # Which person to track for angles
        
    def get_active_connections(self):
        """Get list of all active skeleton connections with their colors and thickness"""
        connections = []
        for group_name, group_config in self.skeleton_groups.items():
            if group_config['enabled'] and self.show_skeleton:
                for conn in self.skeleton_connections[group_name]:
                    connections.append({
                        'indices': conn,
                        'color': group_config['color'],
                        'thickness': group_config['thickness']
                    })
        return connections
    
    def reset_display_defaults(self):
        """Reset display options to default RTMPose values"""
        self.show_bboxes = True
        self.show_track_ids = True
        self.show_keypoints = True
        self.keypoint_size = 3
        self.show_skeleton = True
        
        # Reset skeleton groups to defaults
        self.skeleton_groups = {
            'left_leg': {'enabled': True, 'color': (0, 255, 0), 'thickness': 2},
            'right_leg': {'enabled': True, 'color': (0, 128, 255), 'thickness': 2},
            'left_arm': {'enabled': True, 'color': (0, 255, 0), 'thickness': 2},
            'right_arm': {'enabled': True, 'color': (0, 128, 255), 'thickness': 2},
            'torso': {'enabled': True, 'color': (255, 0, 255), 'thickness': 2},
            'head': {'enabled': True, 'color': (255, 153, 51), 'thickness': 2}
        }
