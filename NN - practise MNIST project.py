import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

(train_image, train_label),( test_image, test_label) = mnist.load_data()
class_name = ['Zero', 'One', 'Two', 'Three','Four','Five','Six','Seven','Eight','Nine']

train_image = train_image.astype('float32') / 255
test_image = test_image.astype('float32') / 255

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(Dense(15, input_shape=(28,28), activation='relu'))
model.add(Dense(12,input_shape=(28,28),activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_image,train_label,epochs=5)

test_loss, test_acc = model.evaluate(test_image,test_label)
print('Accuracy:', test_acc)

prediction = model.predict(test_image)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_image[i])
    plt.title('Prediction:'+ class_name[np.argmax(prediction[i])])
    plt.show()

image = load_img('seven.png')
img_array = img_to_array(image)
prediction2 = model.predict(image)
plt.imshow(image)
plt.title('Prediction' + class_name[prediction2])
plt.show()
