"""
USB Telescope Control Software
This application provides control for USB telescopes with zoom and AI upscaling capabilities.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QPushButton, QSlider, QStatusBar, QFrame,
                            QFileDialog, QMessageBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# Optional: For advanced upscaling - requires additional installation
try:
    import torch
    from torch.nn import functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class UpscalerThread(QThread):
    """Thread for handling image upscaling without blocking the UI"""
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    
    def __init__(self, image, factor):
        super().__init__()
        self.image = image
        self.factor = factor
        
    def run(self):
        """Run the upscaling process"""
        try:
            if HAS_TORCH and self.factor > 1:
                # Using PyTorch for better upscaling if available
                self.progress.emit(10)
                
                # Convert image to tensor
                img_tensor = torch.from_numpy(self.image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                self.progress.emit(30)
                
                # Upscale using bicubic interpolation first
                upscaled = F.interpolate(
                    img_tensor, 
                    scale_factor=self.factor, 
                    mode='bicubic', 
                    align_corners=False
                )
                self.progress.emit(70)
                
                # Convert back to numpy array
                result = (upscaled.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
                self.progress.emit(90)
                
            else:
                # Fallback to OpenCV interpolation
                self.progress.emit(30)
                result = cv2.resize(
                    self.image, 
                    (int(self.image.shape[1] * self.factor), 
                     int(self.image.shape[0] * self.factor)),
                    interpolation=cv2.INTER_CUBIC
                )
                self.progress.emit(70)
                
            # Apply optional image enhancement
            result = self._enhance_image(result)
            self.progress.emit(100)
            
            self.finished.emit(result)
            
        except Exception as e:
            print(f"Error in upscaling: {e}")
            self.finished.emit(self.image)  # Return original on error
    
    def _enhance_image(self, image):
        """Apply some basic enhancement to the upscaled image"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge((l, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        
        return enhanced


class TelescopeControlApp(QMainWindow):
    """Main application for controlling USB telescopes"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USB Telescope Control Software")
        self.setMinimumSize(1024, 768)
        
        # Initialize variables
        self.camera = None
        self.camera_index = 0
        self.available_cameras = []
        self.zoom_factor = 1.0
        self.upscale_factor = 2
        self.current_frame = None
        self.is_recording = False
        self.video_writer = None
        self.is_processing = False
        
        # Setup UI
        self.init_ui()
        
        # Setup camera timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        
        # Scan for cameras on startup
        self.scan_cameras()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create top controls
        controls_layout = QHBoxLayout()
        
        # Camera connection group
        connection_group = QGroupBox("Camera Connection")
        connection_layout = QGridLayout(connection_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        connection_layout.addWidget(QLabel("Select Camera:"), 0, 0)
        connection_layout.addWidget(self.camera_combo, 0, 1)
        
        self.scan_btn = QPushButton("Scan for Cameras")
        self.scan_btn.clicked.connect(self.scan_cameras)
        connection_layout.addWidget(self.scan_btn, 1, 0)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_camera_connection)
        connection_layout.addWidget(self.connect_btn, 1, 1)
        
        controls_layout.addWidget(connection_group)
        
        # Zoom control group
        zoom_group = QGroupBox("Zoom Control")
        zoom_layout = QGridLayout(zoom_group)
        
        self.zoom_label = QLabel("Digital Zoom: 1.0x")
        zoom_layout.addWidget(self.zoom_label, 0, 0, 1, 2)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 50)  # 1.0x to 5.0x
        self.zoom_slider.setValue(10)
        self.zoom_slider.setTickInterval(5)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider, 1, 0, 1, 2)
        
        controls_layout.addWidget(zoom_group)
        
        # Upscaling control group
        upscale_group = QGroupBox("AI Upscaling")
        upscale_layout = QGridLayout(upscale_group)
        
        self.upscale_label = QLabel("Upscale Factor: 2x")
        upscale_layout.addWidget(self.upscale_label, 0, 0, 1, 2)
        
        self.upscale_slider = QSlider(Qt.Horizontal)
        self.upscale_slider.setRange(1, 4)  # 1x to 4x
        self.upscale_slider.setValue(2)
        self.upscale_slider.setTickInterval(1)
        self.upscale_slider.setTickPosition(QSlider.TicksBelow)
        self.upscale_slider.valueChanged.connect(self.update_upscale_factor)
        upscale_layout.addWidget(self.upscale_slider, 1, 0, 1, 2)
        
        self.upscale_btn = QPushButton("Upscale Current View")
        self.upscale_btn.clicked.connect(self.upscale_image)
        self.upscale_btn.setEnabled(False)
        upscale_layout.addWidget(self.upscale_btn, 2, 0, 1, 2)
        
        controls_layout.addWidget(upscale_group)
        
        # Capture control group
        capture_group = QGroupBox("Capture Controls")
        capture_layout = QGridLayout(capture_group)
        
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        capture_layout.addWidget(self.capture_btn, 0, 0)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        capture_layout.addWidget(self.record_btn, 0, 1)
        
        controls_layout.addWidget(capture_group)
        
        main_layout.addLayout(controls_layout)
        
        # Image display area
        self.image_frame = QLabel()
        self.image_frame.setAlignment(Qt.AlignCenter)
        self.image_frame.setFrameStyle(QFrame.StyledPanel)
        self.image_frame.setMinimumHeight(480)
        self.image_frame.setStyleSheet("background-color: #222;")
        main_layout.addWidget(self.image_frame, 1)  # Add stretch factor for proper sizing
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. No camera connected.")
        
        # Set the main widget
        self.setCentralWidget(main_widget)
    
    def scan_cameras(self):
        """Scan for available camera devices"""
        self.available_cameras = []
        self.camera_combo.clear()
        
        # OpenCV camera enumeration
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            
            self.available_cameras.append(index)
            self.camera_combo.addItem(f"Camera #{index}")
            cap.release()
            index += 1
        
        if self.available_cameras:
            self.status_bar.showMessage(f"Found {len(self.available_cameras)} camera(s)")
        else:
            self.status_bar.showMessage("No cameras found")
            QMessageBox.warning(self, "No Cameras", "No camera devices were detected.")
    
    def toggle_camera_connection(self):
        """Connect to or disconnect from the selected camera"""
        if self.camera is None:  # Not connected, try to connect
            try:
                idx = self.camera_combo.currentIndex()
                if idx >= 0 and self.available_cameras:
                    self.camera_index = self.available_cameras[idx]
                    self.camera = cv2.VideoCapture(self.camera_index)
                    
                    if self.camera.isOpened():
                        # Start the update timer
                        self.camera_timer.start(30)  # ~33 fps
                        
                        # Update UI
                        self.connect_btn.setText("Disconnect")
                        self.status_bar.showMessage(f"Connected to Camera #{self.camera_index}")
                        self.camera_combo.setEnabled(False)
                        self.scan_btn.setEnabled(False)
                        self.capture_btn.setEnabled(True)
                        self.record_btn.setEnabled(True)
                        self.upscale_btn.setEnabled(True)
                    else:
                        self.camera = None
                        QMessageBox.critical(self, "Connection Failed", 
                                            "Failed to connect to the selected camera.")
                else:
                    QMessageBox.warning(self, "No Camera Selected", 
                                      "Please select a camera from the dropdown.")
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", 
                                    f"An error occurred while connecting to the camera: {str(e)}")
                self.camera = None
        else:  # Already connected, disconnect
            self.disconnect_camera()
    
    def disconnect_camera(self):
        """Disconnect from the current camera"""
        if self.is_recording:
            self.toggle_recording()  # Stop recording first
            
        if self.camera is not None:
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None
            
            # Update UI
            self.connect_btn.setText("Connect")
            self.status_bar.showMessage("Disconnected from camera")
            self.camera_combo.setEnabled(True)
            self.scan_btn.setEnabled(True)
            self.capture_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
            self.upscale_btn.setEnabled(False)
            
            # Clear the image display
            self.image_frame.clear()
            self.image_frame.setStyleSheet("background-color: #222;")
    
    def update_frame(self):
        """Update the video frame from the camera"""
        if self.camera is None or not self.camera.isOpened():
            return
        
        ret, frame = self.camera.read()
        if not ret:
            self.status_bar.showMessage("Failed to capture frame")
            return
        
        # Store the current frame
        self.current_frame = frame.copy()
        
        # Apply digital zoom if necessary
        if self.zoom_factor > 1.0:
            h, w = frame.shape[:2]
            # Calculate the region of interest (ROI) based on zoom
            roi_w, roi_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            
            # Extract ROI and resize to original dimensions
            roi = frame[y1:y1+roi_h, x1:x1+roi_w]
            frame = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Record video if active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
        
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage and QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        self.image_frame.setPixmap(pixmap.scaled(
            self.image_frame.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def update_zoom(self, value):
        """Update the digital zoom factor"""
        self.zoom_factor = value / 10.0
        self.zoom_label.setText(f"Digital Zoom: {self.zoom_factor:.1f}x")
    
    def update_upscale_factor(self, value):
        """Update the AI upscaling factor"""
        self.upscale_factor = value
        self.upscale_label.setText(f"Upscale Factor: {self.upscale_factor}x")
    
    def capture_image(self):
        """Capture a still image from the current frame"""
        if self.current_frame is None:
            return
        
        try:
            # Choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", 
                os.path.join(os.path.expanduser("~"), 
                             f"telescope_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
                "Images (*.png *.jpg)"
            )
            
            if file_path:
                # Save the image
                cv2.imwrite(file_path, self.current_frame)
                self.status_bar.showMessage(f"Image saved to {file_path}")
                
                # Offer to upscale the saved image
                reply = QMessageBox.question(
                    self, "Upscale Image", 
                    "Would you like to upscale this image now?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.upscale_image(file_path)
        
        except Exception as e:
            QMessageBox.critical(self, "Capture Error", f"Failed to save image: {str(e)}")
    
    def toggle_recording(self):
        """Start or stop video recording"""
        if not self.is_recording:  # Start recording
            try:
                # Choose save location
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Video", 
                    os.path.join(os.path.expanduser("~"), 
                                 f"telescope_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"),
                    "Video (*.avi)"
                )
                
                if file_path:
                    # Get frame dimensions
                    if self.current_frame is not None:
                        h, w = self.current_frame.shape[:2]
                        fps = 30
                        
                        # Create VideoWriter object
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.video_writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
                        
                        self.is_recording = True
                        self.record_btn.setText("Stop Recording")
                        self.status_bar.showMessage("Recording started...")
            
            except Exception as e:
                QMessageBox.critical(self, "Recording Error", 
                                    f"Failed to start recording: {str(e)}")
        
        else:  # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            self.record_btn.setText("Start Recording")
            self.status_bar.showMessage("Recording stopped")
    
    def upscale_image(self, image_path=None):
        """Upscale the current frame or a saved image"""
        if self.is_processing:
            return
        
        try:
            # Determine the source image
            if image_path and os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Failed to load the image")
            elif self.current_frame is not None:
                img = self.current_frame.copy()
            else:
                QMessageBox.warning(self, "Upscaling Error", "No image available to upscale")
                return
            
            self.is_processing = True
            self.upscale_btn.setText("Processing...")
            self.upscale_btn.setEnabled(False)
            self.status_bar.showMessage("Upscaling image... This may take a moment.")
            
            # Start upscaling in a separate thread
            self.upscaler_thread = UpscalerThread(img, self.upscale_factor)
            self.upscaler_thread.finished.connect(self.handle_upscaled_image)
            self.upscaler_thread.progress.connect(self.update_upscale_progress)
            self.upscaler_thread.start()
            
        except Exception as e:
            self.is_processing = False
            self.upscale_btn.setText("Upscale Current View")
            self.upscale_btn.setEnabled(True)
            QMessageBox.critical(self, "Upscaling Error", f"Failed to upscale image: {str(e)}")
    
    @pyqtSlot(int)
    def update_upscale_progress(self, value):
        """Update the upscaling progress in the status bar"""
        self.status_bar.showMessage(f"Upscaling image... {value}% complete")
    
    @pyqtSlot(np.ndarray)
    def handle_upscaled_image(self, result_image):
        """Handle the upscaled image result"""
        try:
            # Choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Upscaled Image", 
                os.path.join(os.path.expanduser("~"), 
                             f"telescope_upscaled_{self.upscale_factor}x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
                "Images (*.png *.jpg)"
            )
            
            if file_path:
                # Save the upscaled image
                cv2.imwrite(file_path, result_image)
                self.status_bar.showMessage(f"Upscaled image saved to {file_path}")
            else:
                self.status_bar.showMessage("Upscaling cancelled")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save upscaled image: {str(e)}")
        
        finally:
            self.is_processing = False
            self.upscale_btn.setText("Upscale Current View")
            self.upscale_btn.setEnabled(True)
    
    def closeEvent(self, event):
        """Clean up resources when the application is closed"""
        self.disconnect_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TelescopeControlApp()
    window.show()
    sys.exit(app.exec_())