import streamlit as st
import numpy as np
from PIL import Image
import time
import tempfile
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE='demo.jpg'
DEMO_VIDEO = 'demo.mp4'
#loading models
eye_model = load_model("eyesdetection.h5")
yawn_model = load_model("created_dataset_yawndetection1.h5")
#haarcascade classifiers
face_cascade = cv2.CascadeClassifier ("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width : 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width : 350px;
        margin-left : -350px;
    }
    </style>

    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Face Mesh Sidebar")
st.sidebar.subheader("Parameters")

@st.cache_resource()
def image_resize(image,width=None, height=None,inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r=width/float(w)
        dim = (int(w*r),height)
    else:
        r=width/float(w)
        dim = (width,int(h*r))
    
    #resize image
    resized = cv2.resize(image,dim,interpolation=inter)

    return resized

app_mode = st.sidebar.selectbox('Select App mode',
                                   ['About App','Run on Video'])

if app_mode == "About App":
    st.title("Face Mesh App")
    st.markdown("In this application we are using **MediaPipe** and **StreamLit** to detect **Drowsiness**")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width : 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width : 350px;
            margin-left : -350px;
        }
        </style>

        """,
        unsafe_allow_html=True,
    )  
    #to add video to page
    st.video('demo.mp4')

    st.markdown(
        '''
        About Me \n
        Hey this is our Major Project\n
        Please check out all the details of this project\n
        Feel free to inform if need any improvements at\n
        123004264@sastra.ac.in \n\n
        '''
    )
            
elif app_mode=="Run on Video":

    st.set_option('deprecation.showfileUploaderEncoding',False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
         st.checkbox("recording",value=True)


    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width : 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width : 350px;
            margin-left : -350px;
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
     

    st.sidebar.markdown('---')
    # max_faces = st.sidebar.number_input('Maximum Number of Faces',value=5,min_value = 1)
    # detection_confidence = st.sidebar.slider('min Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    # tracking_confidence = st.sidebar.slider('min tracking Confidence',min_value=0.0,max_value=1.0,value=0.5)
    # st.sidebar.markdown('---')

    # st.markdown("## Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    ##We get our input video here

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
        
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording part
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output1.mp4',codec,fps_input,(width,height))

    st.sidebar.text("Input video")
    st.sidebar.video(tffile.name)

    fps = 0
    i=0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2,circle_radius=1)

    kpi1,kpi2,kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Yawn Status**")
        kpi1_text = st.markdown("-")
    with kpi2:
        st.markdown("**Eyes Status**")
        kpi2_text = st.markdown("-")
    with kpi3:
        st.markdown("**Frame Count**")
        kpi3_text = st.markdown("0")

    
    st.markdown("<hr>",unsafe_allow_html=True)


    #Face Mesh Predictor
    # with mp_face_mesh.FaceMesh(
    #     ) as face_mesh:
    prevTime = 0
    yscore = 0
    escore=0
    while vid.isOpened():
        i+=1
        ret,frame = vid.read()
        if not ret:
            continue
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
        eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 14,scaleFactor = 1.1)
        mouth = mouth_cascade.detectMultiScale(gray,minNeighbors = 20,scaleFactor = 1.3)

        for (a,b,aw,ah) in mouth:
            mou = frame[b-30:b+ah+10,a-15:a+aw+10]
            try:
                resize = tf.image.resize(mou, (256,256))
                predictiony = yawn_model.predict(np.expand_dims(resize/255, 0))
                #Condition for no yawn
                if predictiony<0.3:
                    kpi1_text.write(f"<h1 style='text-align:left; color:red;'>No Yawn</h1><br><h1 style='color:red;'>{yscore}</h1>",unsafe_allow_html=True)
                    yscore = yscore - 1
                    if (yscore < 0):
                        yscore = 0
                #Condition for yawn
                elif predictiony>0.5:
                    kpi1_text.write(f"<h1 style='text-align:left; color:red;'>Yawn</h1><br><h1 style='color:red;'>{yscore}</h1>",unsafe_allow_html=True)

                    yscore=yscore+1
                    if(yscore > 5):
                        try:
                            sound.play()
                        except:
                            pass
            except:
                pass

        for (x,y,w,h) in eyes:
                eye = frame[y:y+h,x:x+w]
                resize = tf.image.resize(eye, (256,256))
                prediction = eye_model.predict(np.expand_dims(resize/255, 0))
                if prediction >0.3:
                    kpi2_text.write(f"<h1 style=' text-align:left; color:red;'>Eyes Open</h1><br><h1 style='color:red;'>{escore}</h1> ",unsafe_allow_html=True)
                    escore = escore - 1
                    if (escore < 0):
                        escore = 0

        if len(eyes)==0:
            kpi2_text.write(f"<h1 style='text-align:left; color:red;'>Eyes Closed</h1><br><h1 style='color:red;'>{escore}</h1>",unsafe_allow_html=True)
            escore=escore+1
            if(escore > 10):
                try:
                    sound.play()
                except:
                    pass 

        #FPS Counter Logic
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime=currTime

        if record:
            out.write(frame)

        #Dashboard
        kpi3_text.write(f"<h1 style='text-align:left; color:red;'> {int(fps)}</h1> ",unsafe_allow_html=True)

        frame = cv2.resize(frame,(0,0),fx = 0.8,fy=0.8)
        frame = image_resize(image=frame,width=560)
        stframe.image(frame,channels='BGR',use_column_width=True)
