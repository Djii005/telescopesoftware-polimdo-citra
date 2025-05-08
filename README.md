# USB Telescope Control Software

A simple desktop application for controlling USB telescopes with zoom and image enhancement.

## Features

- Connect to any USB camera or telescope
- Adjust digital zoom from 1.0x to 5.0x
- Capture images and record videos
- Upscale and enhance images with AI

## Installation

1. Make sure you have Python installed
2. Install required packages:
   ```
   pip install PyQt5 opencv-python numpy
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage

1. Select your camera from the dropdown and click "Connect"
2. Use the sliders to adjust zoom and upscaling settings
3. Capture images or record video using the buttons
4. Disconnect when finished

## Creating an Executable

To create a standalone .exe file:
```
pip install pyinstaller
pyinstaller --onefile --windowed --name TelescopeControl main.py
```

## Requirements

- Python 3.7+
- PyQt5
- OpenCV
- NumPy
- Optional: PyTorch (for better upscaling)
