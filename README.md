# Vehicle Detection, Tracking, and Counting

This project uses **YOLOv11**, **SORT**, and **OpenCV** to perform vehicle detection, tracking, and counting in video footage.

## Features

- **Vehicle Detection**: Detects cars, trucks, buses, and motorbikes.
- **Object Tracking**: Tracks detected vehicles with unique IDs.
- **Counting**: Counts vehicles that cross a virtual line.
- **Custom Mask**: Focuses on specific regions of the video using a custom mask.

## Requirements

- Python 3.x
- Install dependencies:
    ```bash
    pip install ultralytics opencv-python numpy cvzone
    ```

## Setup

1. Place the following files in the same directory:
   - `cars.mp4` (Input video file)
   - `mask.png` (Region of interest mask)
   - `yolo11n.pt` (YOLOv11 model file)
   
2. Run the script:
    ```bash
    python car-counter.py
    ```

3. The script will process the video, detect, track, and count vehicles that cross the defined line, and save the result in `output.mp4`.

## Customization

- Adjust the **line coordinates** for counting in the `limits` variable.
- Modify the **confidence threshold** to filter vehicle detections.
- Train your own YOLO model if needed.
