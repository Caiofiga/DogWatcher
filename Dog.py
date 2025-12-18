import os
from multiprocessing import Process
import numpy as np
import time
import imutils
import cv2

face_d = cv2.CascadeClassifier(
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Stream URLs from environment variables
RTSP_URL_JARDIM = os.environ.get("RTSP_URL_JARDIM")
RTSP_URL_SALA = os.environ.get("RTSP_URL_SALA")
first_frame = 0
w_size = 1200

mp3file = 'ping.mp3'
cou = 1
ogasdsa = 'n'


def run():
    # Initialize both video captures
    # Require env vars for streams
    if not RTSP_URL_JARDIM:
        raise RuntimeError("Missing RTSP_URL_JARDIM environment variable")
    if not RTSP_URL_SALA:
        raise RuntimeError("Missing RTSP_URL_SALA environment variable")

    rtsp_capture = cv2.VideoCapture(RTSP_URL_JARDIM, cv2.CAP_FFMPEG)

    http_capture = cv2.VideoCapture(RTSP_URL_SALA, cv2.CAP_FFMPEG)

    # Get first frame from RTSP for motion detection
    result, first_frame = rtsp_capture.read()

    while not result or first_frame is None:
        print("Failed to read the first frame from RTSP. retrying.")
        result, first_frame = rtsp_capture.read()
        continue

    first_frame = imutils.resize(first_frame, width=w_size)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
    take_time = time.time()
    main_emer = 0
    con_emer = time.time()

    while True:
        # Read from RTSP stream
        r1, rtsp_img = rtsp_capture.read()
        if not r1 or rtsp_img is None:
            print("Failed to read RTSP frame. Retrying...")
            continue

        # Read from HTTP stream
        r2, http_img = http_capture.read()
        if not r2 or http_img is None:
            print("Failed to read HTTP frame. Retrying...")
            # Continue with just RTSP if HTTP fails
            http_img = None

        # Process RTSP image for motion detection
        rtsp_img = imutils.resize(rtsp_img, width=w_size)
        gray = cv2.cvtColor(rtsp_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frameDel = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frameDel, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts, res = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            if (time.time() - take_time > 3):
                cv2.imwrite(str(time.time()) + ".jpg", rtsp_img)
                take_time = time.time()
            if (time.time() - con_emer > 1):
                main_emer = 0
            else:
                main_emer += time.time() - con_emer
            con_emer = time.time()
            if main_emer > 10:
                first_frame = gray
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(rtsp_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.putText(rtsp_img, str(main_emer), (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

        # Display both windows
        cv2.imshow("RTSP Stream", rtsp_img)

        if http_img is not None:
            http_img = imutils.resize(http_img, width=w_size)
            cv2.imshow("HTTP Stream", http_img)

        if cv2.waitKey(1) == ord("q"):
            break

    # Release both captures
    rtsp_capture.release()
    http_capture.release()


if __name__ == '__main__':
    p = Process(target=run)
    p.start()
    p.join()
