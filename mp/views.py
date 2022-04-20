from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index1(request):
    return HttpResponse('<u>Hello</u>') 

def index2(request):
    return render(request, 'mp/test1.html')    

from tensorflow import keras
from mp import modeltest
from mp import wtsmodel

def mp_model(request):

    actions1 = [
    '오늘',
    '날씨',
    '맑다',
    ]
    model1 = keras.models.load_model("dataset/mediapipe_model.h5")

    mp_words = modeltest.meadia_pipe(model1, actions1)
    
    #print(mp_words) 
    np_words2 = wtsmodel.new_text(mp_words)
    #print(np_words2)

    sentence1 = wtsmodel.predict_mo(np_words2)
    #print(sentence1)  

    return render(request, 'mp/test2.html',{ 'data': sentence1 })

