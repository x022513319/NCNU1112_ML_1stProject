import keras
import random

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from data import load_data
import numpy as np
np.random.seed(10)
batch_size = 128
num_classes = 2
epochs = 5

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)
print("train data:",'images:',x_train.shape," labels:",y_train.shape) 
print("test data:",'images:',x_test.shape," labels:",y_test.shape) 

index = [i for i in range(len(x_train))]
random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

index = [i for i in range(len(x_test))]
random.shuffle(index)
x_test = x_test[index]
y_test = y_test[index]

#x_train = x_train.reshape(203, 112500)
#x_test = x_test.reshape(10, 112500)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0].shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_test.shape)


#model = Sequential()
#model.add(Dense(512, activation='relu', input_shape=(112500,)))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(num_classes, activation='softmax'))
#model.summary()

# Step 2. 建立模型

model = Sequential()

# 卷積層1與池化層1

model.add(Conv2D(filters=16,kernel_size=(3,3),
                 input_shape=(150, 250, 3), 
                 activation='relu', 
                 padding='same'))

model.add(Dropout(rate=0.3))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())
model.add(Dropout(rate=0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['acc'])

train_history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)


import matplotlib.pyplot as plt
def show_train_history(train_type,test_type):
    plt.plot(train_history.history[train_type])
    plt.plot(train_history.history[test_type])
    plt.title('Train History')
    if train_type == 'acc':
        plt.ylabel('Accuracy')
    elif train_type == 'loss':
        plt.ylabel('Loss')
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

scores = model.evaluate(x_test,y_test,verbose=0)
print(scores[:2])

prediction=model.predict(x_test)
prediction = prediction[:10]

label_dict={0:"redlight",1:"greenlight"}
print(label_dict)		


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labelsss,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
        
        title=str(i)+','+label_dict[labelsss[i][1]]
        if len(prediction)>0:
            title+='=>'+label_dict[np.argmax(prediction[i])]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
#print(prediction)
plot_images_labels_prediction(x_test,y_test,prediction,0,10)
#print(y_test)