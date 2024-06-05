# A final run of the network using the test set for validation (instead of
# the validation set.)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.regularizers import l1
import echidna as data
import boundary
import losses

X_train = data.X_train
X_test = data.X_test                   # CHANGED
Y_train = to_categorical(data.Y_train)
Y_test = to_categorical(data.Y_test)   # CHANGED

model = Sequential()
model.add(Dense(100, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(30, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),   # CHANGED
                    epochs=30000, batch_size=25)

boundary.show(model, data.X_test, data.Y_test,          # CHANGED
              title="Test set")
losses.plot(history)
