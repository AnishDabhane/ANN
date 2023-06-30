import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
import tqdm

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)= mnist.load_data()
print(x_train.shape)
x_test.shape
x_train = x_train/ 255
x_test = x_test/ 255

model = Sequential
flattened_array = input_array.flatten()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',matrics=[accuracy])
model.fit(x_train,y_train,validatioin_split=0.2,epochs = 10)

# to make predictions
y_prob = model.predict(x_test)

y_pred=y_prob.argmax(axis=1)
#It's output is an array which contains the digits from 0-9 and 1st element tells us the model's prediction

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
#97.84%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history[''])

plt.imshow(x_test[0],'hot')

model.predict(x_test[0].reshape(1,28,28)).argmax(axis=1)
