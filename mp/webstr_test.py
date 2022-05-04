from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse
import threading
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import os
import statistics

from google.protobuf.json_format import MessageToDict

def strtest(self):

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # actions = name
    seq_length = 20

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    model = load_model("dataset/slt_model.h5")

    actions = ['0','1','2','3','4','5','6','7','8','9','10','가렵다','개','공원','금요일','내년','내일','냄새나다',
        '누나','동생','수화','물','아래','바다','배고프다','병원','불','산','삼키다','선생님','수요일','아빠',
        '아파트','앞','어제','어지러움','언니','엄마','오늘','오른쪽','오빠','올해','왼쪽','월요일','위에',
        '음식물','일요일','자동차','작년','집','친구','택시','토요일','학교','형','화요일','화장실',
        '가다','감사합니다','괜찮습니다','나','남자','내리다','당신','돕다','맞다',
        '모르다','미안합니다','반드시','부탁합니다','빨리','수고','수화','슬프다','싫다',
        '아니다','안녕하세요','알다','없다','여자','오다','있다','잘','좋다','주다','타다',
        '끝', '무엇', '키우다', '우리', '단체', '번역', '만들다', '사랑합니다', '어디']

    #cap = cv2.VideoCapture(0)
    seq, action_seq, sentences = [], [], []
    #Text = ['']
    this_action = ['',0]

    while True:
        this_action[1] += 1
        if this_action[1] == 50:
            this_action[0] = ''
            this_action[1] = 0 
        
        font = ImageFont.truetype("fonts/gulim.ttc", 40)
        self_frame_pil = Image.fromarray(self.frame)
        draw = ImageDraw.Draw(self_frame_pil)
        draw.text((30,50), this_action[0], font=font, fill=(1, 1, 1))
        self.frame = np.array(self_frame_pil) 

        self.grabbed, self.frame = self.video.read()
    
        self.frame = cv2.flip(self.frame, 1) 
        result = hands.process(self.frame)

        if result.multi_hand_landmarks is not None:
            hand_arr = []
            right_hand, left_hand = np.zeros((21,3)), np.zeros((21,3))
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree
                angle_label = np.array(angle, dtype=np.float32)

                handedness_dict = MessageToDict(result.multi_handedness[0])
                if handedness_dict['classification'][0]['label'] == 'Right':
                    right_hand = joint
                else:
                    left_hand = joint

                hand_arr.extend(angle_label)

                mp_drawing.draw_landmarks(self.frame, res, mp_hands.HAND_CONNECTIONS)

            if len(hand_arr) == 15:
                handedness_dict = MessageToDict(result.multi_handedness[0])
                if handedness_dict['classification'][0]['label'] == 'Right':
                    hand_arr = np.concatenate((np.zeros(15), hand_arr))
                else:
                    hand_arr = np.concatenate((hand_arr, np.zeros(15)))
            elif len(hand_arr) > 30:
                continue

            hand_distance = left_hand - right_hand
            hand_distance /= np.linalg.norm(hand_distance, axis=1)[:, np.newaxis]
            hand_arr = np.concatenate((hand_arr, hand_distance.flatten()))
            seq.append(hand_arr)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            if conf < 0.8:
                continue
            
            action = actions[i_pred]
            action_seq.append(action)


            # if len(action_seq) < 3:
            #     continue
            # this_action = '?'
            # if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            #     this_action = action

            # -- 
            if this_action[0] != action:
                this_action[0] = action

            # font = ImageFont.truetype("fonts/gulim.ttc", 40)
            # self_frame_pil = Image.fromarray(self.frame)
            # draw = ImageDraw.Draw(self_frame_pil)
            # draw.text((30,50), this_action, font=font, fill=(1, 1, 1))
            # self.frame = np.array(self_frame_pil)

            # cv2.putText(self.frame, text=this_action,
            #             org=(int(self.frame.shape[1] / 2), 100),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=2,
            #             color=(255, 255, 255),
            #             thickness=3)

        else:
            if len(action_seq) > 0:
                sentences.append(statistics.mode(action_seq))
            action_seq = [] 

        return action_seq
        
        # # cv2.imshow('img', img)
        # if cv2.waitKey(1) == ord('q'):
        #     cv2.destroyAllWindows()
        #     # self.release()
        #     # cv2.destroyAllWindows()
        #     return sentences