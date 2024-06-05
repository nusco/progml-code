from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255
Y_train = to_categorical(Y_train_raw)
Y_test = to_categorical(Y_test_raw)

model = Sequential()
model.add(Dense(1200, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          epochs=100, batch_size=32)
