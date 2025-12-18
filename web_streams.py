from flask import Flask, render_template, Response
import os
import cv2
import imutils
import time
import numpy as np

# Stream URLs are provided via environment variables (set by docker-compose/.env)
RTSP_URL_JARDIM = os.environ.get("RTSP_URL_JARDIM")
HTTP_FEED_URL = os.environ.get("HTTP_FEED_URL")

# Stream display names (can be overridden using environment variables)
STREAM_1_NAME = os.environ.get("STREAM_1_NAME", "Stream 1")
STREAM_2_NAME = os.environ.get("STREAM_2_NAME", "Stream 2")

app = Flask(__name__)

# Global variables
w_size = 600  # Smaller size for web display
first_frame = None
rtsp_capture = None
http_capture = None


def initialize_cameras():
    """Initialize both camera captures"""
    global rtsp_capture, http_capture, first_frame
    # Require that environment variables are set (no hardcoded credentials)
    if not RTSP_URL_JARDIM:
        raise RuntimeError("Missing RTSP_URL_JARDIM environment variable")
    if not HTTP_FEED_URL:
        raise RuntimeError("Missing HTTP_FEED_URL environment variable")

    # Initialize RTSP capture with more robust settings
    rtsp_capture = cv2.VideoCapture(RTSP_URL_JARDIM, cv2.CAP_FFMPEG)

    # Set buffer size to reduce latency
    rtsp_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rtsp_capture.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability

    # Initialize HTTP capture
    http_capture = cv2.VideoCapture(HTTP_FEED_URL)
    http_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get first frame for motion detection
    result, first_frame = rtsp_capture.read()
    if result and first_frame is not None:
        first_frame = imutils.resize(first_frame, width=w_size)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)


# NOTE: ML/YOLO detection removed — this project now streams video only.


def generate_rtsp_frames():
    """Generate RTSP frames with motion detection and dog detection"""
    global first_frame, rtsp_capture, rtsp_dog_count

    take_time = time.time()
    main_emer = 0
    con_emer = time.time()

    while True:
        if rtsp_capture is None:
            break

        ret, frame = rtsp_capture.read()
        if not ret or frame is None:
            continue

        # Resize and process frame
        frame = imutils.resize(frame, width=w_size)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is not None:
            # Motion detection
            frameDel = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frameDel, 30, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            cnts, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in cnts:
                if cv2.contourArea(contour) < 5000:  # Smaller threshold for web
                    continue

                if (time.time() - take_time > 3):
                    take_time = time.time()

                if (time.time() - con_emer > 1):
                    main_emer = 0
                else:
                    main_emer += time.time() - con_emer
                con_emer = time.time()

                if main_emer > 10:
                    first_frame = gray

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Motion overlay text removed per request; rectangles still drawn

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_http_frames():
    """Generate HTTP stream frames with dog detection"""
    global http_capture, http_dog_count

    while True:
        if http_capture is None:
            break

        ret, frame = http_capture.read()
        if not ret or frame is None:
            # Send a blank frame if no data
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "No HTTP Stream", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame = blank_frame
            http_dog_count = 0
        else:
            frame = imutils.resize(frame, width=w_size)

            # Dog detection
            dogs_detected, frame = detect_dogs(frame)
            http_dog_count = dogs_detected

            # Display dog count

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page with both video streams"""
    return render_template('index.html')


@app.route('/rtsp_feed')
def rtsp_feed():
    """RTSP video feed route"""
    return Response(generate_rtsp_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/http_feed')
def http_feed():
    """HTTP video feed route"""
    return Response(generate_http_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dog_counts')
def dog_counts():
    """Get current dog counts as JSON"""
    from flask import jsonify
    return jsonify({
        'rtsp_dogs': rtsp_dog_count,
        'http_dogs': http_dog_count,
        'total_dogs': rtsp_dog_count + http_dog_count
    })


if __name__ == '__main__':
    # Initialize cameras
    initialize_cameras()

    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        # Clean up
        if rtsp_capture:
            rtsp_capture.release()
        if http_capture:
            http_capture.release()
