"""
Droplet Analysis Script for Microscopy Videos

Analyzes droplets illuminated by laser in microscopy videos.
Detects bright droplets (white/green dots) and calculates density statistics.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


class DropletAnalyzer:
    def __init__(self, video_path, start_time, end_time, x_start, x_stop, y_start, y_stop):
        """
        Initialize the droplet analyzer.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            x_start: Left boundary of analysis region (pixels)
            x_stop: Right boundary of analysis region (pixels)
            y_start: Top boundary of analysis region (pixels)
            y_stop: Bottom boundary of analysis region (pixels)
        """
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop

        self.cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None

        # Results - store detailed frame data
        self.frame_data = []  # List of dicts with frame info
        self.frames_analyzed = 0

    def open_video(self):
        """Open the video file and get properties."""
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video opened: {self.video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps:.2f} seconds")

    def validate_roi(self):
        """Validate the region of interest coordinates."""
        if self.x_start < 0 or self.x_stop > self.frame_width:
            raise ValueError(f"x coordinates must be within 0 and {self.frame_width}")
        if self.y_start < 0 or self.y_stop > self.frame_height:
            raise ValueError(f"y coordinates must be within 0 and {self.frame_height}")
        if self.x_start >= self.x_stop:
            raise ValueError("x_start must be less than x_stop")
        if self.y_start >= self.y_stop:
            raise ValueError("y_start must be less than y_stop")

        print(f"\nAnalysis region:")
        print(f"  X: {self.x_start} to {self.x_stop} (width: {self.x_stop - self.x_start})")
        print(f"  Y: {self.y_start} to {self.y_stop} (height: {self.y_stop - self.y_start})")

    def view_frames_interactive(self, start_frame, end_frame, threshold=50, min_area=10, max_area=100):
        """
        Interactive frame viewer to preview frames and ROI before analysis.
        Click and drag to adjust the ROI rectangle.

        Args:
            start_frame: First frame index to view
            end_frame: Last frame index to view
            threshold: Brightness threshold for droplet detection preview
            min_area: Minimum droplet area for detection preview
            max_area: Maximum droplet area for detection preview

        Returns:
            Tuple of (threshold, min_area, max_area) with adjusted detection parameters
        """
        print("\nOpening interactive frame viewer...")
        print("Instructions:")
        print("  - Click 'Select ROI' button, then click and drag on the image to define a new ROI")
        print("  - Or enter exact pixel coordinates in the text boxes and press Enter")
        print("  - Adjust detection parameters (Threshold, Min area, Max area) and press Enter")
        print("  - Adjust start/end times for analysis")
        print("  - Use buttons to navigate frames")
        print("  - Click 'Show Droplets' to preview detection (green circles)")
        print("  - Click 'Show Sum View' to see persistence view (burn-in effect)")
        print("  - Toggle 'Analysis Only' / 'Entire Video' to control what is summed")
        print("  - Click 'Show Processed' to see preprocessed binary image (CLAHE, blur, threshold, morphology)")
        print("  - Click 'Run Analysis' button or close the window to start analysis")

        # Calculate analysis frame range from current time settings
        analysis_start_frame = int(self.start_time * self.fps)
        analysis_end_frame = int(self.end_time * self.fps)
        if analysis_end_frame > end_frame:
            analysis_end_frame = end_frame

        # State for the viewer
        viewer_state = {
            'current_frame': start_frame,
            'start_frame': start_frame,  # Full video range for viewing
            'end_frame': end_frame,      # Full video range for viewing
            'analysis_start_frame': analysis_start_frame,  # Analysis range
            'analysis_end_frame': analysis_end_frame,      # Analysis range
            'roi_selection_mode': False,  # Toggle for ROI selection mode
            'drawing': False,
            'start_x': None,
            'start_y': None,
            'temp_rect': None,
            'show_droplets': False,  # Toggle for showing droplet detection
            'show_sum_view': False,  # Toggle for showing sum of all frames
            'show_processed': False,  # Toggle for showing processed image (CLAHE, blur, threshold, morphology)
            'sum_analysis_only': True,  # True = sum only analysis window, False = sum entire video
            'sum_frame_cache': None,  # Cache for the summed frame
            'sum_frame_cache_full': None,  # Cache for full video sum
            'threshold': threshold,
            'min_area': min_area,
            'max_area': max_area
        }

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.35)

        def calculate_sum_frame(analysis_only=True):
            """
            Calculate the persistence/burn-in sum of frames.
            Only pixels above the threshold are accumulated to avoid background fog.

            Args:
                analysis_only: If True, sum only analysis window (temporal + spatial).
                              If False, sum entire video (all frames, full frame).
            """
            if analysis_only:
                print("Calculating persistence view (analysis window only)... This may take a moment.")
                start_idx = viewer_state['analysis_start_frame']
                end_idx = viewer_state['analysis_end_frame']
            else:
                print("Calculating persistence view (entire video)... This may take a moment.")
                start_idx = viewer_state['start_frame']
                end_idx = viewer_state['end_frame']

            # Initialize accumulator
            sum_frame = None
            frame_count = 0

            # Get current threshold value
            current_threshold = viewer_state['threshold']

            # Read frames in the specified range
            for frame_idx in range(start_idx, end_idx):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()

                if not ret:
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # If analysis_only mode, extract ROI; otherwise use full frame
                if analysis_only:
                    gray = gray[self.y_start:self.y_stop, self.x_start:self.x_stop]

                # Apply threshold: only keep pixels above threshold
                # Create a binary mask where pixels above threshold are True
                mask = gray >= current_threshold

                # Create thresholded version: pixels above threshold keep their value, others are 0
                thresholded = np.where(mask, gray, 0).astype(np.float32)

                # Normalize to 0-1 range (black=0, white=1)
                normalized = thresholded / 255.0

                if sum_frame is None:
                    sum_frame = normalized
                else:
                    sum_frame += normalized

                frame_count += 1

                # Show progress every 50 frames
                if frame_count % 50 == 0:
                    print(f"  Processed {frame_count} frames...", end='\r')

            print(f"  Processed {frame_count} frames... Done!")

            # Scale and convert to displayable format
            if sum_frame is not None and frame_count > 0:
                # Normalize the sum to 0-255 range for display
                # Higher values = more persistence (droplets appeared there more often)
                max_val = np.max(sum_frame)
                if max_val > 0:
                    normalized_display = (sum_frame / max_val * 255).astype(np.uint8)
                else:
                    normalized_display = sum_frame.astype(np.uint8)

                # Convert grayscale to BGR for consistency with main display
                persistence_frame_bgr = cv2.cvtColor(normalized_display, cv2.COLOR_GRAY2BGR)

                # If analysis_only mode, we need to place the ROI back into full frame context
                if analysis_only:
                    # Create a black full-frame image
                    full_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    # Place the ROI sum into the correct position
                    full_frame[self.y_start:self.y_stop, self.x_start:self.x_stop] = persistence_frame_bgr
                    return full_frame, frame_count
                else:
                    return persistence_frame_bgr, frame_count

            return None, 0

        # Display the first frame
        def show_frame(frame_idx):
            """Display a specific frame with ROI overlay."""
            viewer_state['current_frame'] = max(viewer_state['start_frame'],
                                                 min(frame_idx, viewer_state['end_frame'] - 1))

            # Read the current frame first (needed for all modes)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, viewer_state['current_frame'])
            ret, frame = self.cap.read()

            if not ret:
                print(f"Could not read frame {viewer_state['current_frame']}")
                return

            # Check if we should show processed view
            if viewer_state['show_processed']:
                # Get the processed binary image
                _, _, processed_binary = self.detect_droplets(
                    frame,
                    viewer_state['threshold'],
                    viewer_state['min_area'],
                    viewer_state['max_area'],
                    return_processed=True
                )

                # Create a full-frame black image and place the processed ROI
                full_processed = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                full_processed[self.y_start:self.y_stop, self.x_start:self.x_stop] = processed_binary

                # Convert to RGB for display (will show white droplets on black background)
                frame_rgb = cv2.cvtColor(full_processed, cv2.COLOR_GRAY2RGB)

            # Check if we should show sum view
            elif viewer_state['show_sum_view']:
                # Use appropriate cache based on mode
                if viewer_state['sum_analysis_only']:
                    if viewer_state['sum_frame_cache'] is None:
                        viewer_state['sum_frame_cache'], _ = calculate_sum_frame(analysis_only=True)
                        if viewer_state['sum_frame_cache'] is None:
                            print("Error: Could not calculate sum frame")
                            viewer_state['show_sum_view'] = False
                            return
                    frame = viewer_state['sum_frame_cache']
                else:
                    if viewer_state['sum_frame_cache_full'] is None:
                        viewer_state['sum_frame_cache_full'], _ = calculate_sum_frame(analysis_only=False)
                        if viewer_state['sum_frame_cache_full'] is None:
                            print("Error: Could not calculate sum frame")
                            viewer_state['show_sum_view'] = False
                            return
                    frame = viewer_state['sum_frame_cache_full']

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Clear and redraw
            ax.clear()
            ax.imshow(frame_rgb)

            # Draw ROI rectangle
            rect = plt.Rectangle((self.x_start, self.y_start),
                                 self.x_stop - self.x_start,
                                 self.y_stop - self.y_start,
                                 fill=False, edgecolor='red', linewidth=2, label='ROI')
            ax.add_patch(rect)

            # Detect and draw droplets if enabled
            droplet_count = 0
            density_str = ""
            if viewer_state['show_droplets']:
                count, droplet_data = self.detect_droplets(
                    frame,
                    viewer_state['threshold'],
                    viewer_state['min_area'],
                    viewer_state['max_area']
                )
                droplet_count = count

                # Calculate density
                roi_area = (self.x_stop - self.x_start) * (self.y_stop - self.y_start)
                density = droplet_count / roi_area

                # Draw circles around detected droplets with length/width lines
                for droplet in droplet_data:
                    x = droplet['x']
                    y = droplet['y']
                    radius = droplet['radius']
                    length = droplet['length']
                    width = droplet['width']
                    angle = droplet['angle']

                    circle = plt.Circle((x, y), radius * 10, fill=False, edgecolor='lime', linewidth=0.5)
                    ax.add_patch(circle)

                    # Draw length line (major axis) - red
                    # The stored angle now always corresponds to the length/motion direction
                    angle_rad = np.deg2rad(angle)
                    length_dx = (length / 2) * np.cos(angle_rad)
                    length_dy = (length / 2) * np.sin(angle_rad)
                    ax.plot([x - length_dx, x + length_dx],
                           [y - length_dy, y + length_dy],
                           'r-', linewidth=1, alpha=0.8)

                    # Draw width line (minor axis) - blue
                    # Width is perpendicular to the length/motion direction
                    width_angle_rad = angle_rad + np.pi/2
                    width_dx = (width / 2) * np.cos(width_angle_rad)
                    width_dy = (width / 2) * np.sin(width_angle_rad)
                    ax.plot([x - width_dx, x + width_dx],
                           [y - width_dy, y + width_dy],
                           'b-', linewidth=1, alpha=0.8)

                density_str = f" | Density: {density:.2e}/px²"

            # Calculate time and create title
            total_frames_in_video = viewer_state['end_frame'] - viewer_state['start_frame']
            analysis_frames = viewer_state['analysis_end_frame'] - viewer_state['analysis_start_frame']
            if viewer_state['show_processed']:
                # Show processed view info
                time_sec = viewer_state['current_frame'] / self.fps
                frame_in_video = viewer_state['current_frame'] - viewer_state['start_frame'] + 1
                detection_status = f" | Droplets: {droplet_count}{density_str}" if viewer_state['show_droplets'] else ""
                ax.set_title(f"PROCESSED VIEW (CLAHE→Blur→Threshold→Morphology)\n"
                            f"Frame: {frame_in_video}/{total_frames_in_video} | Time: {time_sec:.2f}s{detection_status}",
                            fontsize=14, fontweight='bold')
            elif viewer_state['show_sum_view']:
                # Show sum view info
                detection_status = f" | Droplets: {droplet_count}{density_str}" if viewer_state['show_droplets'] else ""
                if viewer_state['sum_analysis_only']:
                    mode_str = f"Analysis Window Only - {analysis_frames} frames"
                else:
                    mode_str = f"Entire Video - {total_frames_in_video} frames"
                roi_instruction = "Click and drag to define ROI" if viewer_state['roi_selection_mode'] else "Click 'Select ROI' button to adjust ROI"
                ax.set_title(f"PERSISTENCE VIEW ({mode_str}){detection_status}\n"
                            f"{roi_instruction}",
                            fontsize=14, fontweight='bold')
            else:
                # Show regular frame info
                time_sec = viewer_state['current_frame'] / self.fps
                frame_in_video = viewer_state['current_frame'] - viewer_state['start_frame'] + 1
                detection_status = f" | Droplets: {droplet_count}{density_str}" if viewer_state['show_droplets'] else ""
                roi_instruction = "Click and drag to define ROI" if viewer_state['roi_selection_mode'] else "Click 'Select ROI' button to adjust ROI"
                ax.set_title(f"Frame: {frame_in_video}/{total_frames_in_video} | Time: {time_sec:.2f}s{detection_status}\n"
                            f"{roi_instruction}",
                            fontsize=14, fontweight='bold')
            ax.set_xlabel(f"X pixels", fontsize=10)
            ax.set_ylabel(f"Y pixels", fontsize=10)
            ax.legend(loc='upper right')

            # Add text box with info
            roi_width = self.x_stop - self.x_start
            roi_height = self.y_stop - self.y_start
            analysis_start_time = viewer_state['analysis_start_frame'] / self.fps
            analysis_end_time = viewer_state['analysis_end_frame'] / self.fps
            info_text = (f"ROI: ({self.x_start}, {self.y_start}) to ({self.x_stop}, {self.y_stop})\n"
                        f"Size: {roi_width} × {roi_height} px\n"
                        f"Analysis Range: {analysis_start_time:.2f}s to {analysis_end_time:.2f}s\n"
                        f"Analysis Frames: {viewer_state['analysis_start_frame']} to {viewer_state['analysis_end_frame']}\n"
                        f"Detection: T={viewer_state['threshold']}, "
                        f"A=[{viewer_state['min_area']},{viewer_state['max_area']}]")
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')

            fig.canvas.draw_idle()

        # Create frame navigation buttons (top row)
        ax_back10 = plt.axes([0.15, 0.22, 0.08, 0.035])
        ax_back1 = plt.axes([0.24, 0.22, 0.08, 0.035])
        ax_fwd1 = plt.axes([0.33, 0.22, 0.08, 0.035])
        ax_fwd10 = plt.axes([0.42, 0.22, 0.08, 0.035])
        ax_textbox_frame = plt.axes([0.55, 0.22, 0.15, 0.035])

        btn_back10 = Button(ax_back10, '<<< 10')
        btn_back1 = Button(ax_back1, '< 1')
        btn_fwd1 = Button(ax_fwd1, '1 >')
        btn_fwd10 = Button(ax_fwd10, '10 >>>')
        textbox_frame = TextBox(ax_textbox_frame, 'Frame:', initial=str(start_frame))

        # Create ROI coordinate input boxes (second row)
        ax_x_start = plt.axes([0.15, 0.15, 0.10, 0.035])
        ax_x_stop = plt.axes([0.27, 0.15, 0.10, 0.035])
        ax_y_start = plt.axes([0.45, 0.15, 0.10, 0.035])
        ax_y_stop = plt.axes([0.57, 0.15, 0.10, 0.035])

        textbox_x_start = TextBox(ax_x_start, 'X start:', initial=str(self.x_start))
        textbox_x_stop = TextBox(ax_x_stop, 'X stop:', initial=str(self.x_stop))
        textbox_y_start = TextBox(ax_y_start, 'Y start:', initial=str(self.y_start))
        textbox_y_stop = TextBox(ax_y_stop, 'Y stop:', initial=str(self.y_stop))

        # Create detection parameter input boxes (third row)
        ax_threshold = plt.axes([0.15, 0.08, 0.10, 0.035])
        ax_min_area = plt.axes([0.27, 0.08, 0.10, 0.035])
        ax_max_area = plt.axes([0.45, 0.08, 0.10, 0.035])

        textbox_threshold = TextBox(ax_threshold, 'Threshold:', initial=str(threshold))
        textbox_min_area = TextBox(ax_min_area, 'Min area:', initial=str(min_area))
        textbox_max_area = TextBox(ax_max_area, 'Max area:', initial=str(max_area))

        # Create time input boxes and buttons (bottom row)
        ax_roi_button = plt.axes([0.02, 0.02, 0.09, 0.035])
        ax_start_time = plt.axes([0.15, 0.02, 0.10, 0.035])
        ax_end_time = plt.axes([0.27, 0.02, 0.10, 0.035])
        ax_toggle_button = plt.axes([0.38, 0.02, 0.09, 0.035])
        ax_sum_view_button = plt.axes([0.48, 0.02, 0.09, 0.035])
        ax_sum_mode_button = plt.axes([0.58, 0.02, 0.09, 0.035])
        ax_processed_button = plt.axes([0.78, 0.02, 0.10, 0.035])
        ax_run_button = plt.axes([0.68, 0.02, 0.09, 0.035])

        btn_select_roi = Button(ax_roi_button, 'Select ROI')
        textbox_start_time = TextBox(ax_start_time, 'Start (s):', initial=f"{self.start_time:.1f}")
        textbox_end_time = TextBox(ax_end_time, 'End (s):', initial=f"{self.end_time:.1f}")
        btn_toggle_droplets = Button(ax_toggle_button, 'Show Droplets')
        btn_sum_view = Button(ax_sum_view_button, 'Show Sum View')
        btn_sum_mode = Button(ax_sum_mode_button, 'Analysis Only')
        btn_processed = Button(ax_processed_button, 'Show Processed')
        btn_run_analysis = Button(ax_run_button, 'Run Analysis')

        # Mouse event handlers for ROI selection
        def on_mouse_press(event):
            """Handle mouse button press."""
            if event.inaxes != ax or not viewer_state['roi_selection_mode']:
                return
            viewer_state['drawing'] = True
            viewer_state['start_x'] = int(event.xdata)
            viewer_state['start_y'] = int(event.ydata)

        def on_mouse_move(event):
            """Handle mouse movement while drawing."""
            if not viewer_state['drawing'] or event.inaxes != ax:
                return

            # Remove previous temporary rectangle
            if viewer_state['temp_rect'] is not None:
                viewer_state['temp_rect'].remove()

            # Draw temporary rectangle
            current_x = int(event.xdata)
            current_y = int(event.ydata)
            x_min = min(viewer_state['start_x'], current_x)
            y_min = min(viewer_state['start_y'], current_y)
            width = abs(current_x - viewer_state['start_x'])
            height = abs(current_y - viewer_state['start_y'])

            viewer_state['temp_rect'] = plt.Rectangle(
                (x_min, y_min), width, height,
                fill=False, edgecolor='yellow', linewidth=2,
                linestyle='--', label='New ROI'
            )
            ax.add_patch(viewer_state['temp_rect'])
            fig.canvas.draw_idle()

        def on_mouse_release(event):
            """Handle mouse button release."""
            if not viewer_state['drawing'] or event.inaxes != ax:
                viewer_state['drawing'] = False
                return

            viewer_state['drawing'] = False

            # Calculate new ROI coordinates
            end_x = int(event.xdata)
            end_y = int(event.ydata)

            new_x_start = min(viewer_state['start_x'], end_x)
            new_x_stop = max(viewer_state['start_x'], end_x)
            new_y_start = min(viewer_state['start_y'], end_y)
            new_y_stop = max(viewer_state['start_y'], end_y)

            # Validate the new ROI
            if (new_x_stop - new_x_start) < 10 or (new_y_stop - new_y_start) < 10:
                print("ROI too small (minimum 10x10 pixels). Keeping previous ROI.")
                viewer_state['temp_rect'] = None
                show_frame(viewer_state['current_frame'])
                return

            # Update the analyzer's ROI coordinates
            self.x_start = new_x_start
            self.x_stop = new_x_stop
            self.y_start = new_y_start
            self.y_stop = new_y_stop

            # Update text boxes to reflect the new ROI
            textbox_x_start.set_val(str(self.x_start))
            textbox_x_stop.set_val(str(self.x_stop))
            textbox_y_start.set_val(str(self.y_start))
            textbox_y_stop.set_val(str(self.y_stop))

            print(f"ROI updated: ({self.x_start}, {self.y_start}) to ({self.x_stop}, {self.y_stop})")

            # Invalidate analysis-only cache since ROI changed
            viewer_state['sum_frame_cache'] = None

            # Turn off ROI selection mode after successful selection
            viewer_state['roi_selection_mode'] = False
            btn_select_roi.label.set_text('Select ROI')
            print("ROI selection mode: DISABLED")

            # Clear temporary rectangle and redraw
            viewer_state['temp_rect'] = None
            show_frame(viewer_state['current_frame'])

        # Connect mouse events
        fig.canvas.mpl_connect('button_press_event', on_mouse_press)
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        fig.canvas.mpl_connect('button_release_event', on_mouse_release)

        # Button callbacks
        def backward_10(_):
            show_frame(viewer_state['current_frame'] - 10)

        def backward_1(_):
            show_frame(viewer_state['current_frame'] - 1)

        def forward_1(_):
            show_frame(viewer_state['current_frame'] + 1)

        def forward_10(_):
            show_frame(viewer_state['current_frame'] + 10)

        # Callback functions
        def jump_to_frame(text):
            try:
                frame_num = int(text)
                show_frame(frame_num)
            except ValueError:
                print(f"Invalid frame number: {text}")

        def update_x_start(text):
            try:
                new_val = int(text)
                if 0 <= new_val < self.x_stop:
                    self.x_start = new_val
                    print(f"X start updated to {self.x_start}")
                    # Invalidate analysis-only cache since ROI changed
                    viewer_state['sum_frame_cache'] = None
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid X start: must be between 0 and {self.x_stop}")
                    textbox_x_start.set_val(str(self.x_start))
            except ValueError:
                print(f"Invalid X start value: {text}")
                textbox_x_start.set_val(str(self.x_start))

        def update_x_stop(text):
            try:
                new_val = int(text)
                if self.x_start < new_val <= self.frame_width:
                    self.x_stop = new_val
                    print(f"X stop updated to {self.x_stop}")
                    # Invalidate analysis-only cache since ROI changed
                    viewer_state['sum_frame_cache'] = None
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid X stop: must be between {self.x_start} and {self.frame_width}")
                    textbox_x_stop.set_val(str(self.x_stop))
            except ValueError:
                print(f"Invalid X stop value: {text}")
                textbox_x_stop.set_val(str(self.x_stop))

        def update_y_start(text):
            try:
                new_val = int(text)
                if 0 <= new_val < self.y_stop:
                    self.y_start = new_val
                    print(f"Y start updated to {self.y_start}")
                    # Invalidate analysis-only cache since ROI changed
                    viewer_state['sum_frame_cache'] = None
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid Y start: must be between 0 and {self.y_stop}")
                    textbox_y_start.set_val(str(self.y_start))
            except ValueError:
                print(f"Invalid Y start value: {text}")
                textbox_y_start.set_val(str(self.y_start))

        def update_y_stop(text):
            try:
                new_val = int(text)
                if self.y_start < new_val <= self.frame_height:
                    self.y_stop = new_val
                    print(f"Y stop updated to {self.y_stop}")
                    # Invalidate analysis-only cache since ROI changed
                    viewer_state['sum_frame_cache'] = None
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid Y stop: must be between {self.y_start} and {self.frame_height}")
                    textbox_y_stop.set_val(str(self.y_stop))
            except ValueError:
                print(f"Invalid Y stop value: {text}")
                textbox_y_stop.set_val(str(self.y_stop))

        def update_start_time(text):
            try:
                new_val = float(text)
                if 0 <= new_val < self.end_time:
                    self.start_time = new_val
                    print(f"Analysis start time updated to {self.start_time:.2f}s")
                    # Update analysis frame range (not the viewing range)
                    new_analysis_start = int(self.start_time * self.fps)
                    viewer_state['analysis_start_frame'] = new_analysis_start
                    # Invalidate sum frame cache since analysis range changed
                    viewer_state['sum_frame_cache'] = None
                    # Refresh display
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid start time: must be between 0 and {self.end_time:.2f}s")
                    textbox_start_time.set_val(f"{self.start_time:.1f}")
            except ValueError:
                print(f"Invalid start time value: {text}")
                textbox_start_time.set_val(f"{self.start_time:.1f}")

        def update_end_time(text):
            try:
                new_val = float(text)
                max_time = self.total_frames / self.fps
                if self.start_time < new_val <= max_time:
                    self.end_time = new_val
                    print(f"Analysis end time updated to {self.end_time:.2f}s")
                    # Update analysis frame range (not the viewing range)
                    new_analysis_end = int(self.end_time * self.fps)
                    if new_analysis_end > viewer_state['end_frame']:
                        new_analysis_end = viewer_state['end_frame']
                    viewer_state['analysis_end_frame'] = new_analysis_end
                    # Invalidate sum frame cache since analysis range changed
                    viewer_state['sum_frame_cache'] = None
                    # Refresh display
                    show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid end time: must be between {self.start_time:.2f}s and {max_time:.2f}s")
                    textbox_end_time.set_val(f"{self.end_time:.1f}")
            except ValueError:
                print(f"Invalid end time value: {text}")
                textbox_end_time.set_val(f"{self.end_time:.1f}")

        def update_threshold(text):
            try:
                new_val = int(text)
                if 0 <= new_val <= 255:
                    viewer_state['threshold'] = new_val
                    print(f"Threshold updated to {viewer_state['threshold']}")
                    # Invalidate sum caches since threshold affects sum view
                    viewer_state['sum_frame_cache'] = None
                    viewer_state['sum_frame_cache_full'] = None
                    if viewer_state['show_droplets'] or viewer_state['show_sum_view']:
                        show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid threshold: must be between 0 and 255")
                    textbox_threshold.set_val(str(viewer_state['threshold']))
            except ValueError:
                print(f"Invalid threshold value: {text}")
                textbox_threshold.set_val(str(viewer_state['threshold']))

        def update_min_area(text):
            try:
                new_val = int(text)
                if 1 <= new_val < viewer_state['max_area']:
                    viewer_state['min_area'] = new_val
                    print(f"Min area updated to {viewer_state['min_area']}")
                    if viewer_state['show_droplets']:
                        show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid min area: must be between 1 and {viewer_state['max_area']}")
                    textbox_min_area.set_val(str(viewer_state['min_area']))
            except ValueError:
                print(f"Invalid min area value: {text}")
                textbox_min_area.set_val(str(viewer_state['min_area']))

        def update_max_area(text):
            try:
                new_val = int(text)
                if viewer_state['min_area'] < new_val <= 10000:
                    viewer_state['max_area'] = new_val
                    print(f"Max area updated to {viewer_state['max_area']}")
                    if viewer_state['show_droplets']:
                        show_frame(viewer_state['current_frame'])
                else:
                    print(f"Invalid max area: must be between {viewer_state['min_area']} and 10000")
                    textbox_max_area.set_val(str(viewer_state['max_area']))
            except ValueError:
                print(f"Invalid max area value: {text}")
                textbox_max_area.set_val(str(viewer_state['max_area']))

        def toggle_droplets_callback(_):
            # Toggle droplets state
            viewer_state['show_droplets'] = not viewer_state['show_droplets']

            # If turning on droplets, turn off other layers
            if viewer_state['show_droplets']:
                if viewer_state['show_sum_view']:
                    viewer_state['show_sum_view'] = False
                    btn_sum_view.label.set_text('Show Sum View')
                    print("Sum view turned off")
                if viewer_state['show_processed']:
                    viewer_state['show_processed'] = False
                    btn_processed.label.set_text('Show Processed')
                    print("Processed view turned off")

            # Update button and status
            status = "ON" if viewer_state['show_droplets'] else "OFF"
            btn_toggle_droplets.label.set_text('Hide Droplets' if viewer_state['show_droplets'] else 'Show Droplets')
            print(f"Droplet detection display: {status}")
            show_frame(viewer_state['current_frame'])

        def toggle_sum_view_callback(_):
            # Toggle sum view state
            viewer_state['show_sum_view'] = not viewer_state['show_sum_view']

            # If turning on sum view, turn off other layers
            if viewer_state['show_sum_view']:
                if viewer_state['show_droplets']:
                    viewer_state['show_droplets'] = False
                    btn_toggle_droplets.label.set_text('Show Droplets')
                    print("Droplet detection turned off")
                if viewer_state['show_processed']:
                    viewer_state['show_processed'] = False
                    btn_processed.label.set_text('Show Processed')
                    print("Processed view turned off")

            # Update button and status
            status = "ON" if viewer_state['show_sum_view'] else "OFF"
            btn_sum_view.label.set_text('Hide Sum View' if viewer_state['show_sum_view'] else 'Show Sum View')
            print(f"Sum view display: {status}")
            show_frame(viewer_state['current_frame'])

        def toggle_sum_mode_callback(_):
            viewer_state['sum_analysis_only'] = not viewer_state['sum_analysis_only']
            # Update button label
            btn_sum_mode.label.set_text('Analysis Only' if viewer_state['sum_analysis_only'] else 'Entire Video')
            mode = "Analysis Window" if viewer_state['sum_analysis_only'] else "Entire Video"
            print(f"Sum mode: {mode}")
            # Refresh if sum view is currently active
            if viewer_state['show_sum_view']:
                show_frame(viewer_state['current_frame'])

        def toggle_processed_callback(_):
            # Toggle processed view state
            viewer_state['show_processed'] = not viewer_state['show_processed']

            # If turning on processed view, turn off other layers
            if viewer_state['show_processed']:
                if viewer_state['show_droplets']:
                    viewer_state['show_droplets'] = False
                    btn_toggle_droplets.label.set_text('Show Droplets')
                    print("Droplet detection turned off")
                if viewer_state['show_sum_view']:
                    viewer_state['show_sum_view'] = False
                    btn_sum_view.label.set_text('Show Sum View')
                    print("Sum view turned off")

            # Update button and status
            status = "ON" if viewer_state['show_processed'] else "OFF"
            btn_processed.label.set_text('Hide Processed' if viewer_state['show_processed'] else 'Show Processed')
            print(f"Processed view display: {status}")
            show_frame(viewer_state['current_frame'])

        def select_roi_callback(_):
            # Toggle ROI selection mode
            viewer_state['roi_selection_mode'] = not viewer_state['roi_selection_mode']

            # Update button label
            if viewer_state['roi_selection_mode']:
                btn_select_roi.label.set_text('Cancel ROI')
                print("ROI selection mode: ENABLED - Click and drag on image to define new ROI")
            else:
                btn_select_roi.label.set_text('Select ROI')
                print("ROI selection mode: DISABLED")

            # Refresh to update title
            show_frame(viewer_state['current_frame'])

        def run_analysis_callback(_):
            print("\n'Run Analysis' button clicked!")
            print(f"Final settings:")
            print(f"  Time range: {self.start_time:.2f}s to {self.end_time:.2f}s")
            print(f"  ROI: ({self.x_start}, {self.y_start}) to ({self.x_stop}, {self.y_stop})")
            plt.close(fig)

        # Connect callbacks
        btn_back10.on_clicked(backward_10)
        btn_back1.on_clicked(backward_1)
        btn_fwd1.on_clicked(forward_1)
        btn_fwd10.on_clicked(forward_10)
        textbox_frame.on_submit(jump_to_frame)

        textbox_x_start.on_submit(update_x_start)
        textbox_x_stop.on_submit(update_x_stop)
        textbox_y_start.on_submit(update_y_start)
        textbox_y_stop.on_submit(update_y_stop)

        textbox_threshold.on_submit(update_threshold)
        textbox_min_area.on_submit(update_min_area)
        textbox_max_area.on_submit(update_max_area)

        textbox_start_time.on_submit(update_start_time)
        textbox_end_time.on_submit(update_end_time)
        btn_select_roi.on_clicked(select_roi_callback)
        btn_toggle_droplets.on_clicked(toggle_droplets_callback)
        btn_sum_view.on_clicked(toggle_sum_view_callback)
        btn_sum_mode.on_clicked(toggle_sum_mode_callback)
        btn_processed.on_clicked(toggle_processed_callback)
        btn_run_analysis.on_clicked(run_analysis_callback)

        # Show initial frame
        show_frame(start_frame)

        plt.show()

        # Reset video position after viewer closes
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Frame viewer closed. Using ROI: ({self.x_start}, {self.y_start}) to ({self.x_stop}, {self.y_stop})")
        print(f"Detection parameters: threshold={viewer_state['threshold']}, min_area={viewer_state['min_area']}, max_area={viewer_state['max_area']}")
        print("Starting analysis...")

        # Return adjusted detection parameters
        return viewer_state['threshold'], viewer_state['min_area'], viewer_state['max_area']

    def detect_droplets(self, frame, threshold=50, min_area=10, max_area=100, return_processed=False):
        """
        Detect droplets in a frame using enhanced blob detection.
        Measures streak length and width for each droplet.

        Uses preprocessing steps to improve detection:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
        2. Gaussian blur for noise reduction
        3. Binary thresholding
        4. Morphological operations (opening and closing) to clean up binary image

        Args:
            frame: Input frame (BGR format)
            threshold: Brightness threshold for droplet detection
            min_area: Minimum droplet area in pixels
            max_area: Maximum droplet area in pixels
            return_processed: If True, also return the processed binary image

        Returns:
            If return_processed is False:
                Tuple of (count, droplet_data) where droplet_data is a list of dicts containing
                droplet information: x, y, radius, length, width, angle
            If return_processed is True:
                Tuple of (count, droplet_data, processed_image) where processed_image is the
                final binary image after all preprocessing steps
        """
        # Extract the region of interest
        roi = frame[self.y_start:self.y_stop, self.x_start:self.x_stop]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast, making dim droplets more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        # Kernel size (3,3) is small to preserve droplet edges
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Apply threshold to find bright spots
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Optional morphological operations to clean up the binary image
        # Using a 3x3 kernel with 1 iteration of closing to connect pixels that are 1 pixel apart
        # This connects white pixels separated by 1 black pixel (diagonal or orthogonal)
        # Opening is removed to avoid eroding away small droplets
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours (droplets)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and analyze each droplet
        valid_droplets = 0
        droplet_data = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_droplets += 1

                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Calculate equivalent radius
                    radius = int(np.sqrt(area / np.pi))

                    # Measure streak length and width
                    # For contours with >= 5 points, use minAreaRect for accurate measurement
                    # For smaller contours, use bounding box as fallback
                    if len(contour) >= 5:
                        # Get minimum area rectangle to measure streak length and width
                        rect = cv2.minAreaRect(contour)
                        # rect = ((center_x, center_y), (width, height), angle)
                        # width and height are the dimensions of the rotated rectangle
                        # The angle is the rotation of the rectangle from horizontal
                        (_, _), (rect_w, rect_h), angle = rect

                        # Determine which dimension is the length and adjust angle accordingly
                        # The angle from minAreaRect corresponds to rect_w's direction
                        # We need the angle to always correspond to the length direction
                        if rect_w >= rect_h:
                            # rect_w is the length (motion direction)
                            length = rect_w
                            width = rect_h
                            length_angle = angle  # angle already corresponds to length
                        else:
                            # rect_h is the length (motion direction)
                            length = rect_h
                            width = rect_w
                            length_angle = angle + 90  # rotate 90° to get length direction

                        angle = length_angle  # Store angle of length/motion direction
                    else:
                        # Fallback for small contours: use simple bounding box
                        _, _, w_bound, h_bound = cv2.boundingRect(contour)
                        length = max(w_bound, h_bound)
                        width = min(w_bound, h_bound)
                        angle = 0.0  # No rotation info for simple bounding box

                    # Store location in absolute frame coordinates
                    abs_x = cx + self.x_start
                    abs_y = cy + self.y_start

                    droplet_info = {
                        'x': abs_x,
                        'y': abs_y,
                        'radius': radius,
                        'length': length,
                        'width': width,
                        'angle': angle
                    }
                    droplet_data.append(droplet_info)

        if return_processed:
            return valid_droplets, droplet_data, binary
        else:
            return valid_droplets, droplet_data

    def analyze(self, threshold=50, min_area=10, max_area=100, show_progress=True, view_frames=False):
        """
        Analyze the video and detect droplets.

        Args:
            threshold: Brightness threshold for droplet detection
            min_area: Minimum droplet area in pixels
            max_area: Maximum droplet area in pixels
            show_progress: Whether to show progress during analysis
            view_frames: Whether to show interactive frame viewer before analysis
        """
        self.open_video()
        self.validate_roi()

        # Calculate frame range
        start_frame = int(self.start_time * self.fps)
        end_frame = int(self.end_time * self.fps)

        if end_frame > self.total_frames:
            end_frame = self.total_frames
            print(f"\nWarning: end_time exceeds video duration. Using end_frame={end_frame}")

        print(f"\nAnalyzing frames {start_frame} to {end_frame}")
        print(f"Detection parameters:")
        print(f"  Threshold: {threshold}")
        print(f"  Min area: {min_area} pixels")
        print(f"  Max area: {max_area} pixels")
        print()

        # Show interactive frame viewer if requested
        if view_frames:
            # Open viewer with full video range (0 to total_frames)
            # but keep analysis range for time inputs
            threshold, min_area, max_area = self.view_frames_interactive(
                0, self.total_frames, threshold, min_area, max_area
            )
            # Recalculate frame range in case it was changed in the viewer
            start_frame = int(self.start_time * self.fps)
            end_frame = int(self.end_time * self.fps)
            if end_frame > self.total_frames:
                end_frame = self.total_frames

            # Update detection parameters display
            print(f"\nUsing adjusted detection parameters:")
            print(f"  Threshold: {threshold}")
            print(f"  Min area: {min_area} pixels")
            print(f"  Max area: {max_area} pixels")
            print()

        # Set video to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Analyze frames
        self.frame_data = []
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Detect droplets
            count, droplet_data = self.detect_droplets(frame, threshold, min_area, max_area)

            # Store frame data with coordinates and results
            time_sec = current_frame / self.fps
            frame_info = {
                'frame_number': current_frame,
                'time_seconds': time_sec,
                'droplet_count': count,
                'droplets': droplet_data,  # List of dicts with x, y, radius, length, width, angle
                'roi_coordinates': {
                    'x_start': self.x_start,
                    'x_stop': self.x_stop,
                    'y_start': self.y_start,
                    'y_stop': self.y_stop
                }
            }
            self.frame_data.append(frame_info)

            current_frame += 1
            self.frames_analyzed += 1

            if show_progress and self.frames_analyzed % 100 == 0:
                print(f"  Processed {self.frames_analyzed} frames...", end='\r')

        if show_progress:
            print(f"  Processed {self.frames_analyzed} frames... Done!")

        self.cap.release()

    def plot_first_frame(self, save_path=None):
        """
        Plot the first analyzed frame with ROI and annotations.

        Args:
            save_path: Optional path to save the plot. If None, displays interactively.
        """
        if not self.frame_data:
            raise ValueError("No data to plot. Run analyze() first.")

        # Get first frame info
        first_frame_info = self.frame_data[0]
        frame_num = first_frame_info['frame_number']
        time_sec = first_frame_info['time_seconds']
        droplet_count = first_frame_info['droplet_count']
        droplets = first_frame_info.get('droplets', [])

        # Open video and read the first frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_num}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Display frame
        ax.imshow(frame_rgb)

        # Draw ROI rectangle
        rect = plt.Rectangle((self.x_start, self.y_start),
                             self.x_stop - self.x_start,
                             self.y_stop - self.y_start,
                             fill=False, edgecolor='red', linewidth=3, label='Analysis ROI')
        ax.add_patch(rect)

        # Draw circles around detected droplets with length/width annotations
        for i, droplet in enumerate(droplets):
            x = droplet['x']
            y = droplet['y']
            radius = droplet['radius']
            length = droplet['length']
            width = droplet['width']
            angle = droplet['angle']

            label = 'Detected Droplets' if i == 0 else None
            circle = plt.Circle((x, y), radius * 3, fill=False, edgecolor='lime', linewidth=1, label=label)
            ax.add_patch(circle)

            # Draw length line (major axis)
            # The stored angle now always corresponds to the length/motion direction
            angle_rad = np.deg2rad(angle)
            length_dx = (length / 2) * np.cos(angle_rad)
            length_dy = (length / 2) * np.sin(angle_rad)
            ax.plot([x - length_dx, x + length_dx],
                   [y - length_dy, y + length_dy],
                   'r-', linewidth=1.5, alpha=0.7)

            # Draw width line (minor axis, perpendicular to length)
            # Width is perpendicular to the length/motion direction
            width_angle_rad = angle_rad + np.pi/2
            width_dx = (width / 2) * np.cos(width_angle_rad)
            width_dy = (width / 2) * np.sin(width_angle_rad)
            ax.plot([x - width_dx, x + width_dx],
                   [y - width_dy, y + width_dy],
                   'b-', linewidth=1.5, alpha=0.7)

        # Calculate density
        roi_area = (self.x_stop - self.x_start) * (self.y_stop - self.y_start)
        density = droplet_count / roi_area if roi_area > 0 else 0

        # Add title with frame and time info
        ax.set_title(f"First Analyzed Frame\n"
                    f"Frame: {frame_num} | Time: {time_sec:.3f}s | Droplets: {droplet_count} | "
                    f"Density: {density:.2e}/px²",
                    fontsize=14, fontweight='bold', pad=20)

        # Add axes labels
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)

        # Add legend
        ax.legend(loc='upper right', fontsize=11)

        # Add info box with ROI coordinates
        info_text = (
            f"ROI Coordinates:\n"
            f"  X: {self.x_start} → {self.x_stop}\n"
            f"  Y: {self.y_start} → {self.y_stop}\n"
            f"  Area: {(self.x_stop - self.x_start) * (self.y_stop - self.y_start):,} px²"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
               family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFirst frame plot saved to: {save_path}")
        else:
            plt.show()

        return fig, ax

    def calculate_statistics(self):
        """Calculate density statistics."""
        if not self.frame_data:
            raise ValueError("No data to analyze. Run analyze() first.")

        # Calculate area of analysis region
        roi_area = (self.x_stop - self.x_start) * (self.y_stop - self.y_start)

        # Extract counts from frame data
        counts = np.array([frame['droplet_count'] for frame in self.frame_data])

        # Calculate statistics
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Density (droplets per square pixel)
        mean_density = mean_count / roi_area
        std_density = std_count / roi_area

        # Also calculate per 1000 square pixels for easier reading
        mean_density_k = mean_density * 1000
        std_density_k = std_density * 1000

        results = {
            'frames_analyzed': self.frames_analyzed,
            'roi_area_pixels': roi_area,
            'mean_droplet_count': float(mean_count),
            'std_droplet_count': float(std_count),
            'min_droplet_count': int(np.min(counts)),
            'max_droplet_count': int(np.max(counts)),
            'mean_density_per_pixel': float(mean_density),
            'std_density_per_pixel': float(std_density),
            'mean_density_per_1000px': float(mean_density_k),
            'std_density_per_1000px': float(std_density_k)
        }

        return results

    def print_results(self, results):
        """Print analysis results."""
        # Conversion factor: pixels per micron
        pixels_per_um = 0.1878
        roi_area_um2 = results['roi_area_pixels'] / (pixels_per_um ** 2)

        print("\n" + "="*60)
        print("DROPLET ANALYSIS RESULTS")
        print("="*60)
        print(f"Frames analyzed: {results['frames_analyzed']}")
        print(f"Analysis region area: {results['roi_area_pixels']:,} pixels")
        print(f"                      ({roi_area_um2:,.0f} μm²)")
        print()
        print(f"Mean droplet count per frame: {results['mean_droplet_count']:.2f}")
        print(f"Std deviation of count: {results['std_droplet_count']:.2f}")
        print(f"Min droplet count: {results['min_droplet_count']}")
        print(f"Max droplet count: {results['max_droplet_count']}")
        print()
        print(f"Mean density: {results['mean_density_per_1000px']:.4f} droplets per 1000 px²")
        print(f"              ({results['mean_density_per_1000px'] * 1000 / (pixels_per_um ** 2):.4f} droplets per 1000 μm²)")
        print(f"Std density:  {results['std_density_per_1000px']:.4f} droplets per 1000 px²")
        print(f"              ({results['std_density_per_1000px'] * 1000 / (pixels_per_um ** 2):.4f} droplets per 1000 μm²)")
        print("="*60)

    def save_results(self, output_path):
        """Save results to JSON file."""
        results = self.calculate_statistics()

        # Add metadata
        full_results = {
            'video_path': str(self.video_path),
            'analysis_parameters': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'x_start': self.x_start,
                'x_stop': self.x_stop,
                'y_start': self.y_start,
                'y_stop': self.y_stop
            },
            'results': results,
            'frame_data': self.frame_data
        }

        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        return results

    def save_droplet_data_numpy(self, output_path):
        """
        Save per-droplet data to a numpy binary file.

        Creates a structured array with one entry per droplet across all frames.
        Each entry contains: frame_number, droplet_id, x, y, length, width

        Args:
            output_path: Path to save the .npy or .npz file
        """
        if not self.frame_data:
            raise ValueError("No data to save. Run analyze() first.")

        # Collect all droplet data
        all_droplets = []

        for frame_info in self.frame_data:
            frame_num = frame_info['frame_number']
            droplets = frame_info.get('droplets', [])

            for droplet_id, droplet in enumerate(droplets):
                all_droplets.append({
                    'frame': frame_num,
                    'droplet_id': droplet_id,
                    'x': droplet['x'],
                    'y': droplet['y'],
                    'length': droplet['length'],
                    'width': droplet['width'],
                    'angle': droplet['angle']
                })

        # Convert to structured numpy array
        dtype = [('frame', 'i4'), ('droplet_id', 'i4'), ('x', 'f4'), ('y', 'f4'),
                 ('length', 'f4'), ('width', 'f4'), ('angle', 'f4')]

        droplet_array = np.array([(d['frame'], d['droplet_id'], d['x'], d['y'],
                                   d['length'], d['width'], d['angle'])
                                  for d in all_droplets], dtype=dtype)

        # Save to file
        np.save(output_path, droplet_array)
        print(f"\nDroplet data saved to: {output_path}")
        print(f"Total droplets across all frames: {len(droplet_array)}")

        return droplet_array

    def plot_histograms(self, save_path=None):
        """
        Create three histograms:
        1. Droplet count per frame with twin axis for density
        2. Streak length distribution (all droplets)
        3. Streak width distribution (all droplets)

        Args:
            save_path: Optional path to save the plot. If None, displays interactively.
        """
        if not self.frame_data:
            raise ValueError("No data to plot. Run analyze() first.")

        # Collect data
        counts = []
        lengths = []
        widths = []

        for frame_info in self.frame_data:
            counts.append(frame_info['droplet_count'])
            droplets = frame_info.get('droplets', [])

            for droplet in droplets:
                lengths.append(droplet['length'])
                widths.append(droplet['width'])

        # Calculate ROI area for density
        roi_area = (self.x_stop - self.x_start) * (self.y_stop - self.y_start)

        # Conversion factor: pixels per micron
        pixels_per_um = 0.1878
        roi_area_um2 = roi_area / (pixels_per_um ** 2)

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram 1: Droplet count per frame with density twin x-axis
        # Create bins with width 1 for count histogram
        count_bins = np.arange(min(counts), max(counts) + 2, 1) if counts else [0, 1]
        ax1.hist(counts, bins=count_bins, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Droplet Count per Frame', fontsize=12)
        ax1.set_ylabel('Frequency (Number of Frames)', fontsize=12)
        ax1.set_title('Droplet Count Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Twin x-axis for density in droplets/μm²
        ax1_twin = ax1.twiny()
        ax1_xlim = ax1.get_xlim()
        ax1_twin.set_xlim(ax1_xlim[0] / roi_area_um2, ax1_xlim[1] / roi_area_um2)
        ax1_twin.set_xlabel('Density (droplets/μm²)', fontsize=12, color='blue')
        ax1_twin.tick_params(axis='x', labelcolor='blue')

        # Add statistics to plot
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        ax1.axvline(mean_count, color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_count:.1f} ± {std_count:.1f}')
        ax1.legend(fontsize=10)

        # Histogram 2: Streak length with velocity twin x-axis
        # Create bins with width 1 for length histogram
        length_bins = np.arange(int(min(lengths)), int(max(lengths)) + 2, 1) if lengths else [0, 1]
        ax2.hist(lengths, bins=length_bins, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Streak Length (pixels)', fontsize=12)
        ax2.set_ylabel('Frequency (Number of Droplets)', fontsize=12)
        ax2.set_title('Streak Length Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Twin x-axis for velocity in μm/s
        # Velocity = (length in μm) × frame_rate = (length in pixels / pixels_per_um) × fps
        ax2_twin = ax2.twiny()
        velocity_factor = self.fps / pixels_per_um
        ax2_xlim = ax2.get_xlim()
        ax2_twin.set_xlim(ax2_xlim[0] * velocity_factor, ax2_xlim[1] * velocity_factor)
        ax2_twin.set_xlabel('Velocity (μm/s)', fontsize=12, color='darkgreen')
        ax2_twin.tick_params(axis='x', labelcolor='darkgreen')

        # Add statistics
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        ax2.axvline(mean_length, color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_length:.2f} ± {std_length:.2f} px')
        ax2.legend(fontsize=10)

        # Histogram 3: Streak width with μm twin x-axis
        # Create bins with width 1 for width histogram
        width_bins = np.arange(int(min(widths)), int(max(widths)) + 2, 1) if widths else [0, 1]
        ax3.hist(widths, bins=width_bins, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Streak Width at Midpoint (pixels)', fontsize=12)
        ax3.set_ylabel('Frequency (Number of Droplets)', fontsize=12)
        ax3.set_title('Streak Width Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Twin x-axis for width in μm
        ax3_twin = ax3.twiny()
        ax3_xlim = ax3.get_xlim()
        ax3_twin.set_xlim(ax3_xlim[0] / pixels_per_um, ax3_xlim[1] / pixels_per_um)
        ax3_twin.set_xlabel('Width (μm)', fontsize=12, color='darkorange')
        ax3_twin.tick_params(axis='x', labelcolor='darkorange')

        # Add statistics
        mean_width = np.mean(widths)
        std_width = np.std(widths)
        ax3.axvline(mean_width, color='darkorange', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_width:.2f} ± {std_width:.2f} px')
        ax3.legend(fontsize=10)

        # Add overall statistics text
        total_droplets = len(lengths)
        fig.suptitle(f'Droplet Analysis Summary - {self.frames_analyzed} frames analyzed, '
                    f'{total_droplets} total droplets detected',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nHistograms saved to: {save_path}")
        else:
            plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("DROPLET STREAK STATISTICS")
        print("="*60)
        print(f"Total droplets analyzed: {total_droplets}")
        print(f"\nStreak measurements:")
        print(f"  Mean length: {mean_length:.2f} ± {std_length:.2f} pixels")
        print(f"              ({mean_length/pixels_per_um:.2f} ± {std_length/pixels_per_um:.2f} μm)")
        print(f"  Mean width:  {mean_width:.2f} ± {std_width:.2f} pixels")
        print(f"              ({mean_width/pixels_per_um:.2f} ± {std_width/pixels_per_um:.2f} μm)")
        print(f"  Aspect ratio: {mean_length/mean_width:.2f}")
        print(f"\nVelocity (assuming streak length = distance per frame):")
        mean_velocity = (mean_length / pixels_per_um) * self.fps
        std_velocity = (std_length / pixels_per_um) * self.fps
        print(f"  Mean velocity: {mean_velocity:.2f} ± {std_velocity:.2f} μm/s")
        print(f"\nSpatial calibration: {pixels_per_um:.4f} pixels/μm")
        print(f"Temporal resolution: {self.fps:.2f} fps")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze droplets in microscopy videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_droplets.py video.mp4 --start-time 5 --end-time 30 \\
      --x-start 100 --x-stop 500 --y-start 50 --y-stop 400

  python analyze_droplets.py video.mp4 -t 10 30 -x 100 500 -y 50 400 \\
      --threshold 80 --min-area 15 --max-area 150
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('-t', '--time', nargs=2, type=float, metavar=('START', 'END'),
                        help='Start and end time in seconds (e.g., -t 5 30)')
    parser.add_argument('--start-time', type=float, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('-x', '--x-range', nargs=2, type=int, metavar=('START', 'STOP'),
                        help='X coordinate range in pixels (e.g., -x 100 500)')
    parser.add_argument('--x-start', type=int, help='X start coordinate')
    parser.add_argument('--x-stop', type=int, help='X stop coordinate')
    parser.add_argument('-y', '--y-range', nargs=2, type=int, metavar=('START', 'STOP'),
                        help='Y coordinate range in pixels (e.g., -y 50 400)')
    parser.add_argument('--y-start', type=int, help='Y start coordinate')
    parser.add_argument('--y-stop', type=int, help='Y stop coordinate')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Brightness threshold for droplet detection (default: 50)')
    parser.add_argument('--min-area', type=int, default=10,
                        help='Minimum droplet area in pixels (default: 10)')
    parser.add_argument('--max-area', type=int, default=100,
                        help='Maximum droplet area in pixels (default: 100)')
    parser.add_argument('--view-frames', action='store_true',
                        help='Show interactive frame viewer before analysis')
    parser.add_argument('--plot-first-frame', action='store_true',
                        help='Plot the first analyzed frame with annotations after analysis')
    parser.add_argument('-o', '--output', help='Output JSON file for results')

    args = parser.parse_args()

    # Parse time arguments
    if args.time:
        start_time, end_time = args.time
    else:
        if args.start_time is None or args.end_time is None:
            parser.error("Must specify either -t/--time or both --start-time and --end-time")
        start_time = args.start_time
        end_time = args.end_time

    # Parse x coordinates
    if args.x_range:
        x_start, x_stop = args.x_range
    else:
        if args.x_start is None or args.x_stop is None:
            parser.error("Must specify either -x/--x-range or both --x-start and --x-stop")
        x_start = args.x_start
        x_stop = args.x_stop

    # Parse y coordinates
    if args.y_range:
        y_start, y_stop = args.y_range
    else:
        if args.y_start is None or args.y_stop is None:
            parser.error("Must specify either -y/--y-range or both --y-start and --y-stop")
        y_start = args.y_start
        y_stop = args.y_stop

    # Create analyzer
    analyzer = DropletAnalyzer(
        video_path=args.video,
        start_time=start_time,
        end_time=end_time,
        x_start=x_start,
        x_stop=x_stop,
        y_start=y_start,
        y_stop=y_stop
    )

    # Run analysis
    analyzer.analyze(
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        view_frames=args.view_frames
    )

    # Calculate and print results
    results = analyzer.calculate_statistics()
    analyzer.print_results(results)

    # Plot first frame if requested
    if args.plot_first_frame:
        analyzer.plot_first_frame()

    # Save results if output file specified
    if args.output:
        analyzer.save_results(args.output)


if __name__ == '__main__':
    # Testing mode with magic numbers
    # To use command line arguments instead, comment out this block and uncomment main()

    # Test parameters
    TEST_VIDEO = "260115174302.mp4"  # Change this to your video file
    TEST_START_TIME = 5.0  # seconds
    TEST_END_TIME = 10.0   # seconds
    TEST_X_START = 100     # pixels
    TEST_X_STOP = 500      # pixels
    TEST_Y_START = 50      # pixels
    TEST_Y_STOP = 400      # pixels

    # Detection parameters (optional)
    TEST_THRESHOLD = 50
    TEST_MIN_AREA = 10
    TEST_MAX_AREA = 100
    TEST_VIEW_FRAMES = True  # Set to True to preview frames before analysis

    # Create analyzer with test parameters
    analyzer = DropletAnalyzer(
        video_path=TEST_VIDEO,
        start_time=TEST_START_TIME,
        end_time=TEST_END_TIME,
        x_start=TEST_X_START,
        x_stop=TEST_X_STOP,
        y_start=TEST_Y_START,
        y_stop=TEST_Y_STOP
    )

    # Run analysis
    analyzer.analyze(
        threshold=TEST_THRESHOLD,
        min_area=TEST_MIN_AREA,
        max_area=TEST_MAX_AREA,
        view_frames=TEST_VIEW_FRAMES
    )

    # Calculate and print results
    results = analyzer.calculate_statistics()
    analyzer.print_results(results)

    # Plot the first analyzed frame with annotations (shows length/width lines)
    analyzer.plot_first_frame()
    # Or save to file instead:
    # analyzer.plot_first_frame(save_path='first_frame_annotated.png')

    # Create histograms showing droplet count, length, and width distributions
    analyzer.plot_histograms()
    # Or save to file instead:
    # analyzer.plot_histograms(save_path='droplet_histograms.png')

    # Save per-droplet data to numpy binary file
    analyzer.save_droplet_data_numpy('droplet_data.npy')

    # Optionally save results to JSON
    # analyzer.save_results('test_results.json')

    # Uncomment to use command line arguments instead:
    # main()
