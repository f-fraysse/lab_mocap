"""
Gait analysis utilities for treadmill running
Includes footstrike detection and stride segmentation
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


def butterworth_filter(data, cutoff=10, fs=30, order=1):
    """
    Apply Butterworth low-pass filter to data
    
    Args:
        data: 1D array of data to filter
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        filtered_data: Filtered data array
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def compute_velocity(positions, dt=1/30):
    """
    Compute velocity using finite difference
    
    Args:
        positions: 1D array of positions
        dt: Time step between samples (default 1/30 for 30 fps)
        
    Returns:
        velocities: Array of velocities (same length as positions)
    """
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    # Forward difference for first point
    velocities[0] = (positions[1] - positions[0]) / dt
    # Backward difference for last point
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def detect_footstrikes(ankle_x, fs=30, prominence=5, min_spacing=0.25):
    """
    Detect footstrikes from ankle trajectory using minimum peaks of X coordinate
    
    Args:
        ankle_x: Horizontal ankle positions (filtered)
        fs: Sampling frequency in Hz
        prominence: Minimum prominence for peak detection
        min_spacing: Minimum spacing between footstrikes in seconds
        
    Returns:
        footstrike_indices: Array of frame indices where footstrikes occur
    """
    # Find local minima in horizontal position
    # Invert signal to use find_peaks (finds maxima)
    inverted_x = -ankle_x
    
    # Convert min_spacing to samples
    min_distance = int(min_spacing * fs)
    
    # Find peaks (which are minima in original signal)
    peaks, properties = find_peaks(inverted_x, prominence=prominence, distance=min_distance)
    
    return peaks


def segment_strides(data, footstrike_indices, frames=None):
    """
    Segment data into individual strides
    
    Args:
        data: Dictionary with keys 'hip_angles', 'knee_angles', 'ankle_x', 'ankle_y'
        footstrike_indices: Array of footstrike frame indices
        frames: Optional list of video frames
        
    Returns:
        strides: List of dictionaries, each containing one stride's data
    """
    if len(footstrike_indices) < 2:
        return []
    
    strides = []
    for i in range(len(footstrike_indices) - 1):
        start_idx = footstrike_indices[i]
        end_idx = footstrike_indices[i + 1]
        
        stride = {
            'hip_angles': data['hip_angles'][start_idx:end_idx+1],
            'knee_angles': data['knee_angles'][start_idx:end_idx+1],
            'ankle_x': data['ankle_x'][start_idx:end_idx+1],
            'ankle_y': data['ankle_y'][start_idx:end_idx+1],
            'start_frame': start_idx,
            'end_frame': end_idx,
            'stride_number': i + 1
        }
        
        # Add frames if provided
        if frames is not None and len(frames) > end_idx:
            stride['frames'] = frames[start_idx:end_idx+1]
        
        strides.append(stride)
    
    return strides


def normalize_stride(stride_data, num_points=101):
    """
    Normalize stride data to 0-100% (101 points including both endpoints)
    
    Args:
        stride_data: 1D array of data for one stride
        num_points: Number of points in normalized stride (default 101 for 0-100%)
        
    Returns:
        normalized_data: Array of length num_points
    """
    if len(stride_data) < 2:
        return None
    
    # Original time points (0 to 100%)
    original_time = np.linspace(0, 100, len(stride_data))
    
    # Interpolation function
    interp_func = interp1d(original_time, stride_data, kind='cubic', fill_value='extrapolate')
    
    # New time points
    normalized_time = np.linspace(0, 100, num_points)
    
    # Interpolate
    normalized_data = interp_func(normalized_time)
    
    return normalized_data


def plot_ankle_x_with_footstrikes(ankle_x, footstrike_indices, output_dir, recording_name):
    """
    Create plot of ankle X position with footstrike markers
    
    Args:
        ankle_x: Array of ankle X positions (filtered)
        footstrike_indices: Array of footstrike frame indices
        output_dir: Directory to save plots
        recording_name: Name of recording for file naming
    """
    frames = np.arange(len(ankle_x))
    
    # Ankle X position plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, ankle_x, 'purple', linewidth=2, label='Ankle X Position')
    ax.plot(footstrike_indices, ankle_x[footstrike_indices], 'ro', 
            markersize=8, label='Footstrikes')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Ankle X Position (pixels)', fontsize=12)
    ax.set_title('Left Ankle X Position with Footstrikes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{recording_name}_ankle_x.png'), dpi=150)
    plt.close()


def plot_angles_with_footstrikes(hip_angles, knee_angles, footstrike_indices, 
                                  output_dir, recording_name):
    """
    Create plots of hip and knee angles with footstrike markers
    
    Args:
        hip_angles: Array of hip angles
        knee_angles: Array of knee angles
        footstrike_indices: Array of footstrike frame indices
        output_dir: Directory to save plots
        recording_name: Name of recording for file naming
    """
    frames = np.arange(len(hip_angles))
    
    # Hip angle plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, hip_angles, 'b-', linewidth=2, label='Hip Angle')
    ax.plot(footstrike_indices, hip_angles[footstrike_indices], 'ro', 
            markersize=8, label='Footstrikes')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Hip Angle (degrees)', fontsize=12)
    ax.set_title('Left Hip Angle with Footstrikes', fontsize=14, fontweight='bold')
    ax.set_ylim(-30, 100)  # Fixed Y-axis limits
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{recording_name}_hip_angles.png'), dpi=150)
    plt.close()
    
    # Knee angle plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, knee_angles, 'g-', linewidth=2, label='Knee Angle')
    ax.plot(footstrike_indices, knee_angles[footstrike_indices], 'ro', 
            markersize=8, label='Footstrikes')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Knee Angle (degrees)', fontsize=12)
    ax.set_title('Left Knee Angle with Footstrikes', fontsize=14, fontweight='bold')
    ax.set_ylim(-10, 160)  # Fixed Y-axis limits
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{recording_name}_knee_angles.png'), dpi=150)
    plt.close()


def plot_normalized_strides(strides, output_dir, recording_name):
    """
    Create plots of normalized strides (0-100%) with all strides superimposed
    and frame insets from the third stride
    
    Args:
        strides: List of stride dictionaries
        output_dir: Directory to save plots
        recording_name: Name of recording for file naming
    """
    if len(strides) == 0:
        return
    
    # Normalize all strides
    normalized_hip = []
    normalized_knee = []
    
    for stride in strides:
        norm_hip = normalize_stride(stride['hip_angles'])
        norm_knee = normalize_stride(stride['knee_angles'])
        
        if norm_hip is not None and norm_knee is not None:
            normalized_hip.append(norm_hip)
            normalized_knee.append(norm_knee)
    
    if len(normalized_hip) == 0:
        return
    
    stride_percent = np.linspace(0, 100, 101)
    
    # Extract frames from third stride if available
    frame_insets = None
    if len(strides) >= 3 and 'frames' in strides[2]:
        stride_3_frames = strides[2]['frames']
        percentages = [0, 25, 50, 75, 100]
        frame_indices = [int(p * (len(stride_3_frames) - 1) / 100) for p in percentages]
        frame_insets = [stride_3_frames[idx] for idx in frame_indices]
    
    # Hip angle normalized plot with frame insets
    _plot_normalized_with_frames(
        stride_percent, normalized_hip, frame_insets,
        'Hip Angle (degrees)', 'Left Hip Angle - Normalized Strides',
        (-30, 100), output_dir, f'{recording_name}_hip_normalized.png'
    )
    
    # Knee angle normalized plot with frame insets
    _plot_normalized_with_frames(
        stride_percent, normalized_knee, frame_insets,
        'Knee Angle (degrees)', 'Left Knee Angle - Normalized Strides',
        (-10, 160), output_dir, f'{recording_name}_knee_normalized.png'
    )


def _plot_normalized_with_frames(stride_percent, normalized_data, frame_insets,
                                  ylabel, title, ylim, output_dir, filename):
    """
    Helper function to create normalized stride plot with frame insets
    
    Args:
        stride_percent: Array of percentage points (0-100)
        normalized_data: List of normalized stride data arrays
        frame_insets: List of 5 frames at 0%, 25%, 50%, 75%, 100% or None
        ylabel: Y-axis label
        title: Plot title
        ylim: Tuple of (ymin, ymax) for Y-axis limits
        output_dir: Directory to save plot
        filename: Output filename
    """
    import cv2
    from matplotlib.gridspec import GridSpec
    
    # Create figure with extra space at top for frames
    if frame_insets is not None:
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)
        
        # Frame insets subplot
        ax_frames = fig.add_subplot(gs[0])
        ax_frames.axis('off')
        
        # Main plot subplot
        ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all strides
    for i, data in enumerate(normalized_data):
        ax.plot(stride_percent, data, alpha=0.6, linewidth=1.5, 
                label=f'Stride {i+1}')
    
    # Calculate mean and standard deviation
    mean_data = np.mean(normalized_data, axis=0)
    std_data = np.std(normalized_data, axis=0)
    
    # Plot shaded standard deviation area
    ax.fill_between(stride_percent, mean_data - std_data, mean_data + std_data, 
                     color='gray', alpha=0.3, label='±1 SD')
    
    # Plot mean line
    ax.plot(stride_percent, mean_data, 'k-', linewidth=3, label='Mean')
    
    # Add vertical dotted lines at key percentages
    percentages = [0, 25, 50, 75, 100]
    for pct in percentages:
        ax.axvline(x=pct, linestyle=':', color='black', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Stride Cycle (%)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Add frame insets if available
    if frame_insets is not None:
        # Calculate positions for 5 frames
        frame_width = 0.18  # Width of each frame in figure coordinates
        frame_height = 0.12  # Height of each frame
        y_pos = 0.88  # Y position (top of figure)
        x_positions = [0.05, 0.23, 0.41, 0.59, 0.77]  # X positions for 5 frames
        
        for i, (frame, x_pos, pct) in enumerate(zip(frame_insets, x_positions, percentages)):
            # Create inset axes for frame
            ax_inset = fig.add_axes([x_pos, y_pos, frame_width, frame_height])
            ax_inset.axis('off')
            
            # Display frame (already in RGB format)
            ax_inset.imshow(frame)
            
            # Add percentage label below frame
            ax_inset.text(0.5, -0.15, f'{pct}%', 
                         transform=ax_inset.transAxes,
                         ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def analyze_recording(hip_angles, knee_angles, ankle_coords, output_dir, recording_name, fps=30, frames=None):
    """
    Complete analysis pipeline for a recording
    
    Args:
        hip_angles: List of hip angles
        knee_angles: List of knee angles
        ankle_coords: List of (x, y) tuples for ankle positions
        output_dir: Directory to save outputs
        recording_name: Name of recording
        fps: Frame rate (default 30)
        frames: Optional list of video frames for visualization
        
    Returns:
        analysis_results: Dictionary with analysis results
    """
    # Convert to numpy arrays
    hip_angles = np.array(hip_angles)
    knee_angles = np.array(knee_angles)
    ankle_x = np.array([coord[0] for coord in ankle_coords])
    ankle_y = np.array([coord[1] for coord in ankle_coords])
    
    # Filter ankle trajectory
    ankle_x_filtered = butterworth_filter(ankle_x, cutoff=10, fs=fps, order=1)
    ankle_y_filtered = butterworth_filter(ankle_y, cutoff=10, fs=fps, order=1)
    
    # Detect footstrikes using minimum peaks of ankle X coordinate
    footstrike_indices = detect_footstrikes(ankle_x_filtered, fs=fps, 
                                           prominence=5, min_spacing=0.25)
    
    print(f"Detected {len(footstrike_indices)} footstrikes")
    
    # Prepare data dictionary
    data = {
        'hip_angles': hip_angles,
        'knee_angles': knee_angles,
        'ankle_x': ankle_x_filtered,
        'ankle_y': ankle_y_filtered
    }
    
    # Segment into strides (include frames if provided)
    strides = segment_strides(data, footstrike_indices, frames=frames)
    print(f"Segmented into {len(strides)} complete strides")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    if len(footstrike_indices) > 0:
        # Plot ankle X position with footstrikes
        plot_ankle_x_with_footstrikes(ankle_x_filtered, footstrike_indices, 
                                      output_dir, recording_name)
        
        # Plot angles with footstrikes
        plot_angles_with_footstrikes(hip_angles, knee_angles, footstrike_indices, 
                                     output_dir, recording_name)
    
    if len(strides) > 0:
        plot_normalized_strides(strides, output_dir, recording_name)
    
    # Return results
    results = {
        'footstrike_indices': footstrike_indices,
        'num_strides': len(strides),
        'strides': strides
    }
    
    return results
