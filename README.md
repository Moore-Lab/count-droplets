# Droplet Analysis Script for Microscopy Videos

Comprehensive analysis tool for detecting and characterizing droplets in microscopy videos. This script detects droplets illuminated by laser (appearing as bright streaks), measures their motion characteristics, and provides statistical analysis and visualization tools.

## Features

### Core Analysis
- **Temporal Window Selection**: Define start and end time for analysis
- **Spatial ROI Selection**: Define rectangular region of interest
- **Automatic Droplet Detection**: Brightness-based thresholding with contour detection
- **Streak Measurement**: Automatically measures length and width of droplet streaks
- **Density Calculation**: Droplets per square pixel with statistical analysis

### Interactive Viewer
- **Real-time Frame Navigation**: Step through frames with precise controls
- **Dynamic ROI Adjustment**: Click-and-drag or text input for ROI definition
- **Persistence View**: Burn-in visualization showing cumulative droplet positions
- **Detection Preview**: Visual feedback with adjustable parameters
- **Parameter Tuning**: Real-time adjustment of threshold and size filters

### Visualization & Export
- **Annotated Frame Plots**: Shows detected droplets with streak measurements
- **Comprehensive Histograms**: Count, length, and width distributions
- **Numpy Binary Export**: Per-droplet data for further analysis
- **JSON Export**: Full results and metadata

## Installation

Install the required dependencies:

```bash
pip install opencv-python numpy matplotlib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### Test Mode (Recommended for First Use)

1. Edit the magic numbers in the `if __name__ == '__main__'` section:

```python
TEST_VIDEO = "your_video.mp4"
TEST_START_TIME = 5.0  # seconds
TEST_END_TIME = 10.0   # seconds
TEST_X_START = 100     # pixels
TEST_X_STOP = 500
TEST_Y_START = 50
TEST_Y_STOP = 400
TEST_THRESHOLD = 50
TEST_MIN_AREA = 10
TEST_MAX_AREA = 100
TEST_VIEW_FRAMES = True  # Opens interactive viewer
```

2. Run the script:

```bash
python analyze_droplets.py
```

This will open an interactive viewer where you can refine all parameters before running the full analysis.

## Interactive Viewer Guide

When `TEST_VIEW_FRAMES = True`, an interactive viewer opens with the following features:

### Navigation Controls
- **Frame Navigation**: `<<< 10`, `< 1`, `1 >`, `10 >>>` buttons
- **Jump to Frame**: Enter frame number and press Enter
- **Frame Counter**: Shows current frame / total frames

### ROI Adjustment
- **Click and Drag**: Define ROI by clicking and dragging on the image
- **Text Input**: Enter exact pixel coordinates (X start, X stop, Y start, Y stop)
- **Visual Feedback**: Red rectangle shows current ROI

### Analysis Parameters
- **Threshold**: Brightness cutoff for droplet detection (0-255)
- **Min Area**: Minimum droplet size in pixels
- **Max Area**: Maximum droplet size in pixels
- **Time Range**: Adjust start/end times for analysis window

### Visualization Modes

#### Show Droplets
Click "Show Droplets" to preview detection with current parameters:
- **Green circles**: Detected droplet boundaries
- **Red lines**: Streak length (major axis)
- **Blue lines**: Streak width (minor axis, perpendicular to motion)
- **Density display**: Real-time droplets/px² calculation

#### Show Sum View (Persistence/Burn-in)
Click "Show Sum View" to visualize cumulative droplet presence:
- Shows all frames summed together
- Only pixels above threshold are accumulated (background-free)
- Helps identify optimal ROI by showing spatial distribution
- Toggle between "Analysis Only" (current ROI + time range) or "Entire Video"

### Running Analysis
When satisfied with parameters, click **"Run Analysis"** to proceed with full analysis.

## Command Line Usage

For batch processing or automation, use command line mode:

```bash
python analyze_droplets.py VIDEO [OPTIONS]
```

### Required Arguments

- `VIDEO`: Path to video file

### Time Range (required)

Option 1:
```bash
-t START END
--time START END
```

Option 2:
```bash
--start-time START
--end-time END
```

### Spatial Range (required)

X coordinates (option 1):
```bash
-x START STOP
--x-range START STOP
```

X coordinates (option 2):
```bash
--x-start START
--x-stop STOP
```

Y coordinates (option 1):
```bash
-y START STOP
--y-range START STOP
```

Y coordinates (option 2):
```bash
--y-start START
--y-stop STOP
```

### Optional Parameters

Detection sensitivity:
```bash
--threshold VALUE          # Brightness threshold (default: 50)
--min-area VALUE          # Minimum droplet area in pixels (default: 10)
--max-area VALUE          # Maximum droplet area in pixels (default: 100)
```

Visualization:
```bash
--view-frames             # Open interactive viewer before analysis
--plot-first-frame        # Display annotated first frame after analysis
```

Output:
```bash
-o FILE, --output FILE    # Save results to JSON file
```

### Examples

**Basic analysis with interactive viewer:**
```bash
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 --view-frames
```

**Adjust detection sensitivity:**
```bash
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 \
    --threshold 80 --min-area 15 --max-area 150
```

**Save results to JSON:**
```bash
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 \
    -o results.json
```

## Output Files

### 1. Console Output

Real-time information during analysis:

```
Video opened: video.mp4
  Resolution: 1920x1080
  FPS: 30.0
  Total frames: 9000
  Duration: 300.00 seconds

Analysis region:
  X: 100 to 500 (width: 400)
  Y: 50 to 400 (height: 350)

Analyzing frames 150 to 900
  Processed 750 frames... Done!

============================================================
DROPLET ANALYSIS RESULTS
============================================================
Frames analyzed: 750
Analysis region area: 140,000 pixels

Mean droplet count per frame: 45.23 ± 8.67
Min droplet count: 28
Max droplet count: 67

Mean density: 3.23e-04 droplets/px²
============================================================

DROPLET STREAK STATISTICS
============================================================
Total droplets analyzed: 33923
Mean streak length: 12.45 ± 3.21 pixels
Mean streak width: 4.67 ± 1.15 pixels
Length/Width aspect ratio: 2.67
============================================================
```

### 2. Annotated Frame Plot

Shows the first analyzed frame with:
- **Red rectangle**: Analysis ROI
- **Green circles**: Detected droplet boundaries
- **Red lines**: Streak length measurements
- **Blue lines**: Streak width measurements
- **Title bar**: Frame info, time, count, and density

### 3. Histogram Plots

Three histograms in a single figure:

**Histogram 1: Droplet Count Distribution**
- Primary Y-axis (blue): Frequency (number of frames)
- Secondary Y-axis (red): Density (droplets/px²)
- Shows mean ± standard deviation

**Histogram 2: Streak Length Distribution**
- All detected droplets across all frames
- Shows mean ± standard deviation in pixels

**Histogram 3: Streak Width Distribution**
- Width measured at midpoint perpendicular to motion
- All detected droplets across all frames
- Shows mean ± standard deviation in pixels

### 4. Numpy Binary File (`droplet_data.npy`)

Structured array with per-droplet data:

```python
import numpy as np

# Load data
data = np.load('droplet_data.npy')

# Access fields
frames = data['frame']           # Frame number for each droplet
droplet_ids = data['droplet_id'] # ID within frame (0, 1, 2, ...)
x_coords = data['x']             # X position (pixels)
y_coords = data['y']             # Y position (pixels)
lengths = data['length']         # Streak length (pixels)
widths = data['width']           # Streak width (pixels)
angles = data['angle']           # Rotation angle (degrees)

# Example: Get all droplets from frame 100
frame_100_droplets = data[data['frame'] == 100]

# Example: Calculate statistics
mean_length = np.mean(data['length'])
mean_width = np.mean(data['width'])
aspect_ratio = mean_length / mean_width
```

### 5. JSON File (optional, `-o` flag)

Complete results including:
- Video path and analysis parameters
- Statistical summary
- Per-frame data with all droplet measurements
- ROI coordinates for each frame

## Understanding the Measurements

### Droplet Detection
Droplets are detected using:
1. **Thresholding**: Pixels above brightness threshold
2. **Contour Detection**: Connected regions identified
3. **Size Filtering**: Only droplets within min/max area range
4. **Centroid Calculation**: Center position (x, y)

### Streak Measurements
For each detected droplet:
- **Minimum Area Rectangle**: Best-fit rotated rectangle around contour
- **Length**: Longer dimension (motion direction)
- **Width**: Shorter dimension (perpendicular to motion, measured at midpoint)
- **Angle**: Rotation angle of streak
- **Aspect Ratio**: Length/Width ratio indicates motion speed

### Density Calculation
```
Density = Droplet Count / ROI Area
Units: droplets per square pixel (droplets/px²)
```

Displayed in scientific notation (e.g., 3.45e-04 = 0.000345 droplets/px²)

## Parameter Tuning Guide

### Threshold (Brightness Cutoff)
- **Default**: 50 (out of 255)
- **Too High**: Misses dim droplets, underestimates count
- **Too Low**: Detects background noise, overestimates count
- **Tip**: Use "Show Droplets" in interactive viewer to visualize

### Min Area (Noise Filter)
- **Default**: 10 pixels
- **Purpose**: Filters out single bright pixels and small noise
- **Too High**: Misses small droplets
- **Too Low**: Includes noise as false positives

### Max Area (Clump Filter)
- **Default**: 100 pixels
- **Purpose**: Filters out large bright regions or overlapping droplets
- **Too High**: May include multiple droplets as one
- **Too Low**: Misses large droplets

### ROI Selection Tips
1. Use **persistence view** to see cumulative droplet distribution
2. Select ROI that captures consistent illumination
3. Avoid edges where beam intensity varies
4. Exclude areas with background artifacts

## Workflow Recommendations

### Initial Setup
1. Open video in a player to identify analysis region
2. Run script in test mode with `TEST_VIEW_FRAMES = True`
3. Use persistence view to visualize spatial distribution
4. Adjust ROI to capture illuminated region
5. Use "Show Droplets" to tune threshold and area filters

### Quality Control
- Check that all real droplets have **green circles**
- Verify **red lines** align with streak direction
- Confirm **blue lines** show reasonable widths
- Review histograms for expected distributions

### Batch Processing
Once parameters are optimized:
1. Save parameters in test section
2. Process multiple videos with same settings
3. Export numpy files for comparative analysis

## Troubleshooting

### Detection Issues

**No droplets detected:**
- Lower threshold (try 30-40)
- Reduce min_area
- Check ROI is in illuminated region

**Too many false positives:**
- Raise threshold (try 60-80)
- Increase min_area to filter noise
- Use persistence view to identify noise regions

**Streaks look incorrect:**
- Increase threshold to get cleaner contours
- Adjust max_area if droplets are merging
- Check that droplets aren't overlapping

### Performance

**Slow processing:**
- Reduce time range (analyze shorter segments)
- Reduce ROI size (smaller spatial region)
- Disable interactive viewer for batch jobs

**Memory issues:**
- Process shorter time segments
- Reduce video resolution if possible

### File Format Issues

**"Could not open video file":**
- Verify file path is correct
- Check OpenCV supports format (MP4, AVI, MOV usually work)
- Try converting to a standard format

## Technical Details

### Data Structure

Per-frame storage:
```python
{
    'frame_number': int,
    'time_seconds': float,
    'droplet_count': int,
    'droplets': [
        {
            'x': float,          # centroid x-coordinate
            'y': float,          # centroid y-coordinate
            'radius': float,     # equivalent radius
            'length': float,     # streak length (pixels)
            'width': float,      # streak width (pixels)
            'angle': float       # rotation angle (degrees)
        },
        # ... more droplets
    ],
    'roi_coordinates': {
        'x_start': int, 'x_stop': int,
        'y_start': int, 'y_stop': int
    }
}
```

### Algorithms

**Droplet Detection:**
- Grayscale conversion
- Binary thresholding
- External contour detection (OpenCV)
- Moment-based centroid calculation

**Streak Measurement:**
- Minimum area rectangle fitting (OpenCV `minAreaRect`)
- Major/minor axis extraction
- Length = max(width, height)
- Width = min(width, height)

**Persistence View:**
- Frame-by-frame thresholding
- Pixel-wise accumulation (only above threshold)
- Normalization for display

## Citation

If you use this tool in your research, please cite:

```
Droplet Analysis Script for Microscopy Videos
Automated detection and characterization of droplet streaks
https://github.com/yourusername/droplet-analysis
```

## License

[Specify your license here]

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

## Version History

- **v2.0**: Added streak length/width measurement, interactive viewer, persistence view
- **v1.0**: Initial release with basic droplet counting
