from ultralytics import YOLO
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from pytube import YouTube
import os
from collections import defaultdict
from PIL import Image

# Define the class name mapping based on your YOLO model's configuration
CLASS_NAMES = {
    2: "helmet",
    0: "boots",
    3: "human"
}

def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, classes=[0, 2, 3])
    else:
        # Predict the objects in the image using the model
        res = model.predict(image, conf=conf, classes=[0, 2, 3])

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    object_timing = defaultdict(list)  # Stores the detection start times for each object
    object_durations = defaultdict(int)  # Stores total detection time for each object
    object_images = {}  # Stores representative images for each object
    frame_count = 0
    fps = video.get(cv2.CAP_PROP_FPS)
    detection_threshold = 3  # seconds

    # Load the YOLO model
    model = YOLO("best.pt")  # Change to your model path

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # Run YOLO model on the frame
        results = model(frame, classes=[0, 2, 3])

        # Process detections
        for result in results:
            detections = result.boxes.data
            for detection in detections:
                class_id = int(detection[5])  # Class ID
                confidence = float(detection[4])  # Confidence score

                # Using class_id to identify objects
                if confidence > 0.5:  # Filtering by confidence
                    object_timing[class_id].append(current_time)

                    # Store representative image for the first detection
                    if class_id not in object_images:
                        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
                        cropped_img = frame[y1:y2, x1:x2]
                        object_images[class_id] = cropped_img

    # Calculate object detection durations
    for class_id, timings in object_timing.items():
        # Group consecutive timings within a certain threshold (like 1 second)
        last_time = None
        start_time = None

        for timing in timings:
            if last_time is None or timing - last_time > 1:
                if start_time is not None:
                    duration = last_time - start_time
                    if duration >= detection_threshold:
                        object_durations[class_id] += duration
                start_time = timing

            last_time = timing

        # Finalize the last segment
        if start_time is not None:
            duration = last_time - start_time
            if duration >= detection_threshold:
                object_durations[class_id] += duration

    # Check if any human detected without helmet or boots for 3 seconds
    if object_durations:
        if (2 in object_durations and object_durations[2] >= detection_threshold) or \
           (0 in object_durations and object_durations[0] >= detection_threshold) or \
           (3 in object_durations and object_durations[3] >= detection_threshold):
            # Save the frame of the human without proper PPE
            cv2.imwrite("violation_frame.jpg", object_images[3])  # Assuming class_id for human is 3

    video.release()

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
