import sys
import os
import cv2
import csv
import time
import threading
import serial
import serial.tools.list_ports
import random
from datetime import datetime
from ultralytics import YOLO

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QCheckBox, 
                             QLineEdit, QPushButton, QFrame, QSlider, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont

# --- Port Selector Function ---
def get_default_port():
    """Detects OS and finds the most likely Arduino port."""
    if sys.platform.startswith('win'):
        # Windows
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'Arduino' in p.description:
                return p.device
        return ports[0].device if ports else "COM3"
    else:
        # Linux/macOS
        if os.path.exists('/dev/ttyACM0'):
            return '/dev/ttyACM0'
        elif os.path.exists('/dev/ttyUSB0'):
            return '/dev/ttyUSB0'
        return '/dev/ttyACM0'

# --- Global Logic Variables ---
stop_threads = False
current_lat = 0.0
current_lon = 0.0
current_sats = 0
gps_lock = threading.Lock()
csv_file = 'detection_log.csv'
DEFAULT_PORT = get_default_port()

# --- Backend Logic (GPS & Model) ---

def read_gps(use_mock, serial_port):
    global current_lat, current_lon, current_sats, stop_threads
    
    # Mock GPS
    if use_mock:
        print("Using Mock GPS Data...")
        base_lat = 16.495906
        base_lon = 80.496290
        while not stop_threads:
            with gps_lock:
                current_lat = base_lat + (random.uniform(-0.0005, 0.0005))
                current_lon = base_lon + (random.uniform(-0.0005, 0.0005))
                current_sats = random.randint(5, 12)
            time.sleep(1.0)
        return

    # Real GPS
    try:
        ser = serial.Serial(serial_port, 9600, timeout=1)
        print(f"Connected to GPS at {serial_port}")
        ser.reset_input_buffer()
    except Exception as e:
        print(f"GPS Connection Error: {e}")
        return

    while not stop_threads:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "LAT:" in line and "LON:" in line:
                    parts = line.split(',')
                    with gps_lock:
                        for part in parts:
                            part = part.strip()
                            if "LAT:" in part:
                                try: current_lat = float(part.split(':')[1])
                                except: pass
                            elif "LON:" in part:
                                try: current_lon = float(part.split(':')[1])
                                except: pass
                            elif "SAT:" in part:
                                try: current_sats = int(float(part.split(':')[1]))
                                except: pass
        except Exception:
            continue
    if ser.is_open: ser.close()

def select_model(crop):
    crop = crop.lower()
    models = {
        'tomato': './TomatoE10.pt', 'cotton': './cotton.pt',
        'chilli': './chilli.pt', 'turmeric': './turmeric.pt', 'rose': './rose.pt'
    }
    path = models.get(crop)
    if path: return YOLO(path)
    return None

# --- Worker Thread ---

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object) 
    update_stats_signal = pyqtSignal(float, float, int)

    def __init__(self, crop, model, threshold):
        super().__init__()
        self.crop = crop
        self.model = model
        self.threshold = threshold
        self.running = True
        self.save_cooldown = 2.0
        self.last_save_time = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.running = False
            return

        healthy_defs = {'tomato': ['healthy'], 'cotton': ['Healthy Leaf'], 
                        'rose': ['Healthy', 'rose'], 'chilli': ['Healthy Chilies', 'Healthy Leaves'], 
                        'turmeric': ['healthy_leaf']}
        healthy_keys = healthy_defs.get(self.crop, ['healthy'])

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Inference
                results = self.model(frame, verbose=False)
                result = results[0]
                names = result.names
                
                detected_diseases = []
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    detected_diseases = [names[i] for i, prob in enumerate(probs.data) if prob >= self.threshold]
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if float(box.conf[0]) >= self.threshold:
                            cls_id = int(box.cls[0])
                            detected_diseases.append(names[cls_id])
                    detected_diseases = list(set(detected_diseases))

                # Log to CSV
                current_time = time.time()
                if detected_diseases and (current_time - self.last_save_time > self.save_cooldown):
                    for disease in detected_diseases:
                        if disease not in healthy_keys:
                            with gps_lock:
                                lat, lon, sats = current_lat, current_lon, current_sats
                            
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(csv_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([ts, self.crop, disease, sats, lat, lon])
                            self.last_save_time = current_time
                            print(f"Logged: {disease}")

                # Prepare Image
                annotated_frame = result.plot()
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = qt_img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                
                self.change_pixmap_signal.emit(p)
                self.update_stats_signal.emit(current_lat, current_lon, current_sats)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# --- GUI Window ---

class RecorderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop Disease Recorder")
        self.resize(1000, 700)
        self.is_recording = False
        
        # Apply Global Dark Theme Styling
        self.apply_dark_theme()

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Left Side: Video ---
        video_container = QFrame()
        video_container.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; border: 1px solid #333;")
        video_layout = QVBoxLayout(video_container)
        
        self.video_label = QLabel("Camera Off")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #666; font-size: 18px;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_container, stretch=3)

        # --- Right Side: Controls ---
        controls_frame = QFrame()
        controls_frame.setFixedWidth(320)
        controls_frame.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border-radius: 10px; }
            QLabel { color: #ddd; font-size: 14px; }
        """)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(15)
        main_layout.addWidget(controls_frame)

        # Header
        title = QLabel("CONFIGURATION")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #888; letter-spacing: 1px;")
        controls_layout.addWidget(title)

        # Crop Selection
        controls_layout.addWidget(QLabel("Select Crop Type:"))
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(["tomato", "cotton", "chilli", "turmeric", "rose"])
        self.crop_combo.setStyleSheet("""
            QComboBox { background-color: #3b3b3b; color: white; padding: 8px; border: 1px solid #555; border-radius: 4px; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #aaa; margin-right: 10px; }
        """)
        controls_layout.addWidget(self.crop_combo)

        # GPS Settings
        gps_group = QFrame()
        gps_group.setStyleSheet("background-color: #333; border-radius: 6px; padding: 5px;")
        gps_layout = QVBoxLayout(gps_group)
        
        self.mock_gps_check = QCheckBox("Use Mock GPS")
        self.mock_gps_check.setChecked(True)
        self.mock_gps_check.setStyleSheet("QCheckBox { color: #ccc; }")
        self.mock_gps_check.toggled.connect(self.toggle_gps_entry)
        gps_layout.addWidget(self.mock_gps_check)

        gps_layout.addWidget(QLabel("Serial Port:"))
        self.port_entry = QLineEdit(DEFAULT_PORT)
        gps_layout.addWidget(self.port_entry)

        self.toggle_gps_entry()
        
        controls_layout.addWidget(gps_group)

        # Threshold
        controls_layout.addWidget(QLabel("Confidence Threshold:"))
        
        thresh_layout = QHBoxLayout()
        self.thresh_label = QLabel("0.50")
        self.thresh_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(10, 100)
        self.thresh_slider.setValue(50)
        self.thresh_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; }
            QSlider::handle:horizontal { background: #4CAF50; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.thresh_slider.valueChanged.connect(self.update_thresh_label)
        
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_label)
        controls_layout.addLayout(thresh_layout)

        # Live Stats
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #444; max-height: 1px;")
        controls_layout.addWidget(line)
        
        stats_title = QLabel("LIVE SENSOR DATA")
        stats_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        stats_title.setStyleSheet("color: #888; letter-spacing: 1px;")
        controls_layout.addWidget(stats_title)

        stats_style = "font-family: Consolas; font-size: 13px; color: #00E5FF;"
        self.lbl_lat = QLabel("Lat:  0.000000")
        self.lbl_lat.setStyleSheet(stats_style)
        self.lbl_lon = QLabel("Lon:  0.000000")
        self.lbl_lon.setStyleSheet(stats_style)
        self.lbl_sats = QLabel("Sats: 0")
        self.lbl_sats.setStyleSheet(stats_style)
        
        controls_layout.addWidget(self.lbl_lat)
        controls_layout.addWidget(self.lbl_lon)
        controls_layout.addWidget(self.lbl_sats)

        controls_layout.addStretch()

        # Status Label
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #888; font-style: italic; margin-bottom: 5px;")
        controls_layout.addWidget(self.lbl_status)

        # Single Toggle Button
        self.btn_action = QPushButton("START RECORDING")
        self.btn_action.setFixedHeight(50)
        self.btn_action.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_action.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_button_style("start")
        self.btn_action.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.btn_action)

        # Threads
        self.video_thread = None
        self.gps_thread = None

    def apply_dark_theme(self):
        # Set window background
        self.setStyleSheet("QMainWindow { background-color: #121212; }")

    def set_button_style(self, state):
        if state == "start":
            self.btn_action.setText("START RECORDING")
            self.btn_action.setStyleSheet("""
                QPushButton { background-color: #2e7d32; color: white; border: none; border-radius: 6px; }
                QPushButton:hover { background-color: #388e3c; }
                QPushButton:pressed { background-color: #1b5e20; }
            """)
        else: # stop
            self.btn_action.setText("STOP RECORDING")
            self.btn_action.setStyleSheet("""
                QPushButton { background-color: #c62828; color: white; border: none; border-radius: 6px; }
                QPushButton:hover { background-color: #d32f2f; }
                QPushButton:pressed { background-color: #b71c1c; }
            """)

    def toggle_gps_entry(self):
        is_mock = self.mock_gps_check.isChecked()
        self.port_entry.setEnabled(not is_mock)
           
        if is_mock:
            self.port_entry.setStyleSheet("background-color: #3b3b3b; color: #777; border: 1px solid #444; padding: 5px; border-radius: 4px;")
        else:
            self.port_entry.setStyleSheet("background-color: #222; color: #fff; border: 1px solid #555; padding: 5px; border-radius: 4px;")

    def update_thresh_label(self):
        val = self.thresh_slider.value() / 100.0
        self.thresh_label.setText(f"{val:.2f}")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        global stop_threads
        
        self.lbl_status.setText("Loading Model...")
        QApplication.processEvents()

        crop = self.crop_combo.currentText()
        model = select_model(crop)
        
        if not model:
            QMessageBox.critical(self, "Error", "Could not load model file!")
            self.lbl_status.setText("Error")
            return

        # Start GPS
        stop_threads = False
        self.gps_thread = threading.Thread(target=read_gps, args=(self.mock_gps_check.isChecked(), self.port_entry.text()))
        self.gps_thread.daemon = True
        self.gps_thread.start()

        # Start Video
        threshold = self.thresh_slider.value() / 100.0
        self.video_thread = VideoThread(crop, model, threshold)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_stats_signal.connect(self.update_stats)
        self.video_thread.start()

        # UI Update
        self.is_recording = True
        self.set_button_style("stop")
        self.lbl_status.setText("● Recording Active")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        # CSV Init
        try:
            with open(csv_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'crop', 'disease', 'satellites', 'lat', 'long'])
        except FileExistsError: pass

    def stop_recording(self):
        global stop_threads
        
        stop_threads = True
        if self.video_thread:
            self.video_thread.stop()
        
        # UI Update
        self.is_recording = False
        self.set_button_style("start")
        self.lbl_status.setText("Stopped")
        self.lbl_status.setStyleSheet("color: #888; font-style: italic;")
        self.video_label.clear()
        self.video_label.setText("Camera Off")

    @pyqtSlot(object)
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    @pyqtSlot(float, float, int)
    def update_stats(self, lat, lon, sats):
        self.lbl_lat.setText(f"Lat:  {lat:.6f}")
        self.lbl_lon.setText(f"Lon:  {lon:.6f}")
        self.lbl_sats.setText(f"Sats: {sats}")

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RecorderWindow()
    window.show()
    sys.exit(app.exec())
