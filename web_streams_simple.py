from flask import Flask, render_template, Response
import os
import cv2
import imutils
import time
import numpy as np

# Stream URLs from environment (set these in docker-compose/.env)
RTSP_URL_JARDIM = os.environ.get("RTSP_URL_JARDIM")
RTSP_URL_SALA = os.environ.get("RTSP_URL_SALA")

app = Flask(__name__)

# Global variables
w_size = 600  # Smaller size for web display
first_frame = None
rtsp_capture = None
rtsp2_capture = None


def initialize_cameras():
    """Initialize both RTSP camera captures"""
    global rtsp_capture, rtsp2_capture, first_frame

    # Require environment variables (no hardcoded credentials)
    if not RTSP_URL_JARDIM:
        raise RuntimeError("Missing RTSP_URL_JARDIM environment variable")
    if not RTSP_URL_SALA:
        raise RuntimeError("Missing RTSP_URL_SALA environment variable")

    # Initialize first RTSP capture (Jardim)
    rtsp_capture = cv2.VideoCapture(RTSP_URL_JARDIM, cv2.CAP_FFMPEG)

    # Set buffer size to reduce latency and improve stability
    rtsp_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rtsp_capture.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
    rtsp_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    rtsp_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize second RTSP capture (Sala)
    rtsp2_capture = cv2.VideoCapture(RTSP_URL_SALA, cv2.CAP_FFMPEG)
    rtsp2_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rtsp2_capture.set(cv2.CAP_PROP_FPS, 15)
    rtsp2_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # Get first frame for motion detection
    rtsp2_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    result, first_frame = rtsp_capture.read()
    if result and first_frame is not None:
        first_frame = imutils.resize(first_frame, width=w_size)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)


def generate_rtsp_frames():
    """Generate RTSP frames with motion detection"""
    global first_frame, rtsp_capture

    take_time = time.time()
    main_emer = 0
    con_emer = time.time()

    while True:
        if rtsp_capture is None:
            break

        ret, frame = rtsp_capture.read()
        if not ret or frame is None:
            print("RTSP connection lost, attempting to reconnect...")
            # Try to reconnect
            rtsp_capture.release()
            rtsp_capture = cv2.VideoCapture(RTSP_URL_JARDIM, cv2.CAP_FFMPEG)
            rtsp_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rtsp_capture.set(cv2.CAP_PROP_FPS, 15)
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

            cv2.putText(frame, f"Motion: {main_emer:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_rtsp2_frames():
    """Generate second RTSP stream frames"""
    global rtsp2_capture

    while True:
        if rtsp2_capture is None:
            break

        ret, frame = rtsp2_capture.read()
        if not ret or frame is None:
            print("RTSP2 connection lost, attempting to reconnect...")
            # Try to reconnect
            rtsp2_capture.release()
            rtsp2_capture = cv2.VideoCapture(RTSP_URL_SALA, cv2.CAP_FFMPEG)
            rtsp2_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rtsp2_capture.set(cv2.CAP_PROP_FPS, 15)
            continue

        # Resize frame
        frame = imutils.resize(frame, width=w_size)

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
    """Second RTSP video feed route"""
    return Response(generate_rtsp2_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
        if rtsp2_capture:
            rtsp2_capture.release()
