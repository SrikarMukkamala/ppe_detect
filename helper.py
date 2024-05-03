from ultralytics import YOLO
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from pytube import YouTube
import pandas as pd
import os
from collections import defaultdict
from io import BytesIO
import base64
from PIL import Image

import settings


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
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


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



#def play_webcam(conf, model):
#    source_webcam = settings.WEBCAM_PATH
#    is_display_tracker, tracker = display_tracker_options()
#    if st.sidebar.button('Detect Objects'):
#        try:
#            vid_cap = cv2.VideoCapture(source_webcam)
#            st_frame = st.empty()
#            while (vid_cap.isOpened()):
#                success, image = vid_cap.read()
#                if success:
#                    _display_detected_frames(conf,
#                                             model,
#                                             st_frame,
#                                             image,
#                                             is_display_tracker,
#                                             tracker,
#                                             )
#                else:
#                    vid_cap.release()
#                    break
#        except Exception as e:
#            st.sidebar.error("Error loading video: " + str(e))

def play_webcam(conf, model):
    is_display_tracker, tracker = display_tracker_options()
    st_frame = st.empty()
    try:
        class VideoProcessor:
            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                _display_detected_frames(conf,
                                         model,
                                         st_frame,
                                         image,
                                         is_display_tracker,
                                         tracker,
                                         )
                return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(key="srikar", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

    except Exception as e:
        st.sidebar.error("Error loading video: " + str(e))

            


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# Define the class name mapping based on your YOLO model's configuration
CLASS_NAMES = {
    2: "helmet",
    0: "boots",
    3: "human"
}
# Function to process video and collect object detection data
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
        results = model(frame,classes = [0,2,3])

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

    video.release()
    return object_durations, object_images

def report():
    # Streamlit Application
    st.title("YOLO Model Detection Analysis")

    # File uploader to load the video
    video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mkv", "mov"])

    if video_file:
        # Save the uploaded file temporarily
        video_path = os.path.join("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())

        # Process the video and get object detection durations and images
        with st.spinner("Processing video..."):
            object_durations, object_images = process_video(video_path)

        # Display the results in a DataFrame
        if object_durations:
            df = pd.DataFrame(list(object_durations.items()), columns=["Class ID", "Duration (seconds)"])
            df = df[df["Duration (seconds)"] >= 3]
            # Add class names based on the class ID
            df["Class Name"] = df["Class ID"].map(CLASS_NAMES)

            st.dataframe(df)
            st.success("Processing complete!")
            
            # Display representative images for each object ID
            for class_id in df["Class ID"]:
                if class_id in object_images:
                    img = object_images[class_id]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    pil_img = Image.fromarray(img_rgb)  # Convert to PIL Image

                    # Create an image widget
                    st.image(pil_img, caption=f"Class ID: {class_id}")

        else:
            st.warning("No objects detected for more than 3 seconds.")

        # Cleanup temporary video file
        os.remove(video_path)
    else:
        st.warning("Please upload a video file to analyze.")
