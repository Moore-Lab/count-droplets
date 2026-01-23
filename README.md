# Droplet Analysis Script

This script analyzes droplets illuminated by laser in microscopy videos. It detects bright droplets (appearing as white/green dots) and calculates density statistics within a defined region of interest.

## Features

- Define temporal analysis window (start and end time)
- Define spatial analysis region (rectangular ROI)
- Automatic droplet detection using brightness thresholding
- Calculate droplet density and standard deviation
- Export results to JSON format
- Configurable detection parameters

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400
```

This analyzes `video.mp4` from 5 to 30 seconds, within the rectangular region from x=100-500, y=50-400.

### Detailed Options

```bash
python analyze_droplets.py VIDEO [OPTIONS]
```

**Required arguments:**
- `VIDEO`: Path to the video file

**Time range (one of the following):**
- `-t START END` or `--time START END`: Start and end time in seconds
- `--start-time` and `--end-time`: Specify times separately

**X coordinate range (one of the following):**
- `-x START STOP` or `--x-range START STOP`: X coordinate range in pixels
- `--x-start` and `--x-stop`: Specify coordinates separately

**Y coordinate range (one of the following):**
- `-y START STOP` or `--y-range START STOP`: Y coordinate range in pixels
- `--y-start` and `--y-stop`: Specify coordinates separately

**Optional detection parameters:**
- `--threshold VALUE`: Brightness threshold for droplet detection (default: 50)
  - Increase if detecting too many false positives
  - Decrease if missing dim droplets
- `--min-area VALUE`: Minimum droplet area in pixels (default: 10)
- `--max-area VALUE`: Maximum droplet area in pixels (default: 100)

**Output:**
- `-o FILE` or `--output FILE`: Save results to JSON file

### Examples

**Analyze with default parameters:**
```bash
python analyze_droplets.py my_video.avi -t 10 60 -x 0 800 -y 0 600
```

**Adjust detection sensitivity:**
```bash
python analyze_droplets.py my_video.avi -t 10 60 -x 0 800 -y 0 600 \
    --threshold 80 --min-area 15 --max-area 150
```

**Save results to file:**
```bash
python analyze_droplets.py my_video.avi -t 10 60 -x 0 800 -y 0 600 \
    -o results.json
```

## Output

The script provides:

1. **Console output** with:
   - Video information (resolution, FPS, duration)
   - Analysis region details
   - Processing progress
   - Statistical results

2. **JSON output** (if `-o` specified) containing:
   - All analysis parameters
   - Statistical results (mean, std deviation, min, max)
   - Droplet counts for each frame

### Example Output

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
Detection parameters:
  Threshold: 50
  Min area: 10 pixels
  Max area: 100 pixels

  Processed 750 frames... Done!

============================================================
DROPLET ANALYSIS RESULTS
============================================================
Frames analyzed: 750
Analysis region area: 140,000 pixels

Mean droplet count per frame: 45.23
Std deviation of count: 8.67
Min droplet count: 28
Max droplet count: 67

Mean density: 0.3231 droplets per 1000 px²
Std density: 0.0619 droplets per 1000 px²
============================================================
```

## Adjusting Parameters

### Finding the Right Threshold
- Start with the default (50)
- If you see too many false positives (noise detected as droplets), increase the threshold
- If legitimate droplets are being missed, decrease the threshold

### Setting Area Bounds
- `--min-area`: Filters out noise (single bright pixels)
- `--max-area`: Filters out large bright regions that aren't individual droplets
- Based on your description (10-100 pixels per droplet), the defaults should work well
- Adjust based on your actual droplet sizes

## Tips

1. **Finding the right region:** Open your video in a player, note the pixel coordinates of your analysis region
2. **Skipping empty frames:** Set `start_time` to when droplets first appear
3. **Avoiding dead zones:** Ensure your x/y coordinates exclude areas without laser illumination
4. **Optimizing detection:** Run a few test analyses with different thresholds to find optimal settings

## Troubleshooting

- **"Could not open video file"**: Check the file path and ensure OpenCV supports the video format
- **Too many/few droplets detected**: Adjust `--threshold`, `--min-area`, and `--max-area` parameters
- **Out of bounds error**: Ensure x/y coordinates are within the video frame dimensions
