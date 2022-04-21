import numpy as np
import json
from tensorflow import keras
#from pyjosa.josa import Josa # 조사 오픈소스
    
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)     

def predict_mo(lst):
    reconstructed_model = keras.models.load_model("WTS_model.h5")
    char_indices = json.load(open("char_indices.json","r"))
    indices_char = json.load(open("indices_char.json","r"))   
    
    result =''
    
    for i in lst:
        if i[-1] == '요' or i[-1] == '다':
            result += i + ' '
            continue
            
        x = np.zeros((1, 5, len(char_indices)))
        x[0, len(i), char_indices[i]] = 1.

        preds = reconstructed_model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[str(next_index)]
        
        
        #y = josa_fuc(i, next_char)
        result += (i + next_char + ' ')
        
    return result

def new_text(text):
    try:
        text.remove('')
        while '?' in text:
            text.remove('?')
    except:
        return text
    return text

#mp_words = ['', '?', '오늘', '?', '날씨', '?', '맑다']

#np_words2 = new_text(mp_words)
#print(np_words2)

#sentence1 = predict_mo(np_words2)
#print(sentence1)