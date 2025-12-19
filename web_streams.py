import numpy as np
import time
import imutils
import cv2
from flask import Flask, render_template, Response
import os
import threading


# Stream URLs are provided via environment variables (set by docker-compose/.env)
STREAM_1_URL = os.environ.get("STREAM_1_URL")
STREAM_2_URL = os.environ.get("STREAM_2_URL")

# Stream display names (can be overridden using environment variables)
STREAM_1_NAME = os.environ.get("STREAM_1_NAME", "Stream 1")
STREAM_2_NAME = os.environ.get("STREAM_2_NAME", "Stream 2")

app = Flask(__name__)

# Global variables
w_size = 600  # Smaller size for web display

# Shared frame buffers for multi-user access
feed1_frame = None
feed2_frame = None
feed1_lock = threading.Lock()
feed2_lock = threading.Lock()

# Camera capture objects (used only by background threads)
feed1_capture = None
feed2_capture = None

# Background thread control
capture_threads_running = False


def initialize_cameras():
    """Initialize both camera captures with low-latency settings"""
    global feed1_capture, feed2_capture
    # Require that environment variables are set (no hardcoded credentials)
    if not STREAM_1_URL:
        raise RuntimeError("Missing STREAM_1_URL environment variable")
    if not STREAM_2_URL:
        raise RuntimeError("Missing STREAM_2_URL environment variable")

    feed1_capture = cv2.VideoCapture(STREAM_1_URL, cv2.CAP_FFMPEG)

    # Minimize buffer to reduce latency (keep only 1 frame in buffer)
    feed1_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Don't try to force resolution/FPS - let camera provide native stream

    feed2_capture = cv2.VideoCapture(STREAM_2_URL, cv2.CAP_FFMPEG)
    feed2_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def capture_feed1():
    """Background thread that continuously captures Stream 1 frames"""
    global feed1_frame, feed1_capture, capture_threads_running

    while capture_threads_running:
        if feed1_capture is None:
            time.sleep(0.1)
            continue

        ret, frame = feed1_capture.read()

        if not ret or frame is None:
            # Create a blank "No Stream" frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Stream", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            frame = imutils.resize(frame, width=w_size)

        # Update the shared frame buffer with thread safety
        with feed1_lock:
            feed1_frame = frame.copy()

        time.sleep(0.01)  # Small delay to prevent excessive CPU usage


def capture_feed2():
    """Background thread that continuously captures Stream 2 frames"""
    global feed2_frame, feed2_capture, capture_threads_running

    while capture_threads_running:
        if feed2_capture is None:
            time.sleep(0.1)
            continue

        ret, frame = feed2_capture.read()

        if not ret or frame is None:
            # Create a blank "No Stream" frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Stream", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            frame = imutils.resize(frame, width=w_size)

        # Update the shared frame buffer with thread safety
        with feed2_lock:
            feed2_frame = frame.copy()

        time.sleep(0.01)  # Small delay to prevent excessive CPU usage


def generateFeed1Frames():
    """Generate Stream 1 frames from shared buffer - thread-safe for multiple viewers."""
    global feed1_frame

    while True:
        # Get the latest frame from the shared buffer with thread safety
        with feed1_lock:
            if feed1_frame is None:
                # Create a blank "Initializing" frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Initializing...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = feed1_frame.copy()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


def generateFeed2Frames():
    """Generate Stream 2 frames from shared buffer - thread-safe for multiple viewers."""
    global feed2_frame

    while True:
        # Get the latest frame from the shared buffer with thread safety
        with feed2_lock:
            if feed2_frame is None:
                # Create a blank "Initializing" frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Initializing...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = feed2_frame.copy()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Main page with both video streams"""
    return render_template('index.html', stream1_name=STREAM_1_NAME, stream2_name=STREAM_2_NAME)


@app.route('/rtsp_feed')
def rtsp_feed():
    """RTSP video feed route"""
    return Response(generateFeed1Frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/http_feed')
def http_feed():
    """HTTP video feed route"""
    return Response(generateFeed2Frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Initialize cameras
    initialize_cameras()

    # Start background capture threads
    capture_threads_running = True
    thread1 = threading.Thread(target=capture_feed1, daemon=True)
    thread2 = threading.Thread(target=capture_feed2, daemon=True)
    thread1.start()
    thread2.start()

    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        # Clean up
        capture_threads_running = False
        thread1.join(timeout=2)
        thread2.join(timeout=2)

        if feed1_capture:
            feed1_capture.release()
        if feed2_capture:
            feed2_capture.release()
