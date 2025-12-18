import numpy as np
import time
import imutils
import cv2
from flask import Flask, render_template, Response
import os


# Stream URLs are provided via environment variables (set by docker-compose/.env)
STREAM_1_URL = os.environ.get("STREAM_1_URL")
STREAM_2_URL = os.environ.get("STREAM_2_URL")

# Stream display names (can be overridden using environment variables)
STREAM_1_NAME = os.environ.get("STREAM_1_NAME", "Stream 1")
STREAM_2_NAME = os.environ.get("STREAM_2_NAME", "Stream 2")

app = Flask(__name__)

# Global variables
w_size = 600  # Smaller size for web display
first_frame = None
feed1_capture = None
feed2_capture = None


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


def generateFeed1Frames():
    """Generate Stream 1 frames - simple and reliable."""
    global feed1_capture

    while True:
        if feed1_capture is None:
            break

        ret, frame = feed1_capture.read()

        if not ret or frame is None:
            # send a blank frame instead of blocking
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Stream", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            frame = imutils.resize(frame, width=w_size)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generateFeed2Frames():
    """Generate Stream 2 frames - simple and reliable."""
    global feed2_capture

    while True:
        if feed2_capture is None:
            break

        ret, frame = feed2_capture.read()

        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Stream", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            frame = imutils.resize(frame, width=w_size)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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

    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        # Clean up
        if feed1_capture:
            feed1_capture.release()
        if feed2_capture:
            feed2_capture.release()
