from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index1(request):
    return HttpResponse('<u>Hello</u>') 

def home(request):
    return render(request, 'mp/test3.html')    


## ------------------ 웹캠 부분

from tensorflow import keras
from keras.models import load_model
from mp import modeltest # 모델 테스트 함수
from mp import wtsmodel
from mp.models import WToS 

def mp_model(request):

    name = ['0','1','2','3','4','5','6','7','8','9','10','가렵다','개','공원','금요일','내년','내일','냄새나다',
        '누나','동생','목요일','물','아래','바다','배고프다','병원','불','산','삼키다','선생님','수요일','아빠',
        '아파트','앞','어제','어지러움','언니','엄마','오늘','오른쪽','오빠','올해','왼쪽','월요일','위에',
        '음식물','일요일','자동차','작년','집','친구','택시','토요일','학교','형','화요일','화장실',
        '가다','감사합니다','괜찮습니다','나','남자','내리다','당신','돕다','맞다',
        '모르다','미안합니다','반드시','부탁합니다','빨리','수고','수화','슬프다','싫다',
        '아니다','안녕하세요','알다','없다','여자','오다','있다','잘','좋다','주다','타다',
        '끝', '무엇', '키우다', '우리', '단체', '번역', '만들다', '사랑합니다', '어디']

    model = load_model("dataset/model_0429_1320.h5")

    mp_words = modeltest.meadia_pipe(model, name)
    
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


# -------- 웹 스트리밍 부분 (아직 테스트 중--)

from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse
import threading
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

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
        Text = [''] # 모션 텍스트
        seq_length = 30
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        seq = []
        action_seq = []
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        actions = [
            'today',
            'weather',
            'clear',
            ]
        model = load_model("dataset/mediapipe_model.h5")

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

                if len(hand_arr) == 99:
                    hand_arr.extend(np.zeros(99))

                if len(hand_arr) > 198:
                    continue

                seq.append(hand_arr)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                if conf < 0.9:
                    continue

                #print(i_pred)
                action = actions[i_pred]
                action_seq.append(action)
                if len(action_seq) < 3:
                    continue
                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                cv2.putText(self.frame, text=this_action,
                            org=(int(self.frame.shape[1] / 2), 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(255, 255, 255),
                            thickness=3)
                # font = ImageFont.truetype("fonts/gulim.ttc", 20)
                # self.frame = Image.fromarray(self.frame)
                # draw = ImageDraw.Draw(self.frame)
                # draw.text((30,50), this_action, font=font, fill=(0,0,255))
                # self.frame = np.array(self.frame)

                if Text[-1] != this_action :
                    Text.append(this_action)

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




