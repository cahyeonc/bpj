# 모델링

import numpy as np
from tensorflow import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# seq_날씨_1650334424.npy
# seq_맑다_1650334424.npy
# seq_오늘_1650334403.npy
# seq_오늘_1650334424.npy

data = np.concatenate([
    np.load('dataset/seq_오늘_1650334424.npy'),
    np.load('dataset/seq_오늘_1650334403.npy'),
    np.load('dataset/seq_맑다_1650334424.npy'),
    np.load('dataset/seq_날씨_1650334424.npy')], axis=0)


x = data[:,:,:-1]
y = data[:,0,-1]


y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

clear_session()

model = Sequential()
model.add(LSTM(256, input_shape=(30, 198)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[-1], activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0, min_delta=0)

model.fit(x_train, y_train, validation_split=0.2, epochs=100, callbacks=[es], verbose=0)

y_pred = model.predict(x_test)
accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

model.save("mediapipe_model.h5")