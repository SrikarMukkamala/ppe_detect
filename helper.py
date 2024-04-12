from ultralytics import YOLO
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from pytube import YouTube

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
