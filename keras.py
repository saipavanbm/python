import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
dataset=tensorflow.keras.datasets.mnist.load_data(path="mnist.npz")
train,test=dataset
x_train,y_train=train
x_test,y_test=test
x_test.shape
x_train_1d = x_train.reshape(-1 , 28*28)
x_test_1d = x_test.reshape(-1 , 28*28)
x_train = x_train_1d.astype('float32')
x_test = x_test_1d.astype('float32')
x_train = x_train / 255.0
x_test = x_test/ 255.0
input_shape = (28, 28, 1)
y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
h = model.fit(x_train, y_train_cat, epochs=1)
accuracy=model.evaluate(x_train,y_train_cat)
accuracy=accuracy[1]*100
import os
file = open("accuracy.txt","w+")
file.write(str(accuracy))
file.close()
os.system("mv /accuracy.txt /pyth/")
model.save('mnist.h5')





