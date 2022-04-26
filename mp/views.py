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
        self.video = cv2.VideoCapture(0)
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
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera): # 영상 바이너리 코드를 실시간으로 처리
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    

@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        #cam = cv2.VideoCapture(0)
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

