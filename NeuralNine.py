import os
# to load images and process them computer vision library is used
import cv2
import numpy as np
# just for data visualization
import matplotlib.pyplot as plt
# for ML part
import tensorflow as tf

 
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()

# #normalizing brightness in between 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# converts the above layer into single 1-D array instead of Grid
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(70, activation='relu'))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7)

model.save("handwritten.model")

#insted of training model everytime, simply load it 
model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test,y_test)

print(loss)
print(accuracy)


image_number =1
while os.path.isfile(f"digits/img{image_number}.png"):
    try:
        image = cv2.imread(f"digits/img{image_number}.png") 
        image = np.invert(np.array([img]))
        predicion = model.predict(img)
        print(f"This digit id probably a {np.argmax(prediction)}")
        plt.imshow(image[0],camp=plt.cm.binary)
        plt.show()
    except:
        print("Error!!!")    