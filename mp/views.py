from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index1(request):
    return HttpResponse('<u>Hello</u>') 

def home(request):
    return render(request, 'mp/test3.html')    




# --------

from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse
import threading
import cv2
import mediapipe as mp
import numpy as np

class VideoCamera(object):

    def __init__(self): # 초기선언, 환경
        self.video = cv2.VideoCapture(0) # 카메라에서 정보를 받아옴
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        # 영상을 jpeg로 인코딩하여, yield로 호출한 URL로 실시간으로 바이너리를 전송
        return jpeg.tobytes()

    def update(self): # 영상 실시간 처리
        seq_length = 30
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        seq = []
        action_seq = []
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        while True:
            self.grabbed, self.frame = self.video.read()
            #img0 = self.frame.copy()
    
            self.frame = cv2.flip(self.frame, 1) # 1 좌우반전 0 상하반전
            #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            result = hands.process(self.frame)
            #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                hand_arr = []
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    angle = np.degrees(angle) 
                    d = np.concatenate([joint.flatten(), angle])
                    hand_arr.extend(d)
                    mp_drawing.draw_landmarks(self.frame, res, mp_hands.HAND_CONNECTIONS)

            #cv2.imshow('img', self.frame)

def gen(self): # 영상 바이너리 코드를 실시간으로 처리
    while True:
        frame = self.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
                # 웹 Header 의 mineType을 multipart/x-mixed-replace 으로 선언    
    except:
        print("error")
        pass


## ----------------------------------
from tensorflow import keras
from keras.models import load_model
from mp import modeltest
from mp import wtsmodel
from mp.models import WToS 

def mp_model(request):

    actions1 = [
    '오늘',
    '날씨',
    '맑다',
    ]
    model1 = load_model("dataset/mediapipe_model.h5")

    mp_words = modeltest.meadia_pipe(model1, actions1)
    
    #print(mp_words) 
    np_words2 = wtsmodel.new_text(mp_words)
    #print(np_words2)

    sentence1 = wtsmodel.predict_mo(np_words2)
    #print(sentence1)  

    WToS(text=sentence1).save()

    return render(request, 'mp/test2.html',{ 'data': sentence1 })#, 'text' : WToS.text })

def show(request):
    wtos = WToS.objects.all()
    return render(request, 'mp/test1.html', { 'data': wtos }) 

