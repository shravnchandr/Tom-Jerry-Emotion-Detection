import pandas
import numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import PIL
from PIL import Image

min_max_scaler = MinMaxScaler()

def make_train():
    baseheight = 90
    base_folder = 'Training Frames Pixels/'
    pixelList = []

    for i in range(297):
        img = Image.open(base_folder+ 'trainingFrame' +str(i) +'.jpg')
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
        pixels = numpy.array(img).ravel()
        pixelList.append(pixels)

    x_train = numpy.array(pixelList)
    
    return x_train


def make_test():
    baseheight = 90
    base_folder = 'Testing Frames/'
    pixelList = []

    for i in range(186):
        img = Image.open(base_folder+ 'testingFrame' +str(i) +'.jpg')
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
        pixels = numpy.array(img).ravel()
        pixelList.append(pixels)

    x_test = numpy.array(pixelList)
    
    return x_test


x_train = make_train()
x_train = min_max_scaler.fit_transform(x_train)

x_test = make_test()
x_test = min_max_scaler.fit_transform(x_test)


training_dataset = pandas.read_csv('Train.csv')

temp = training_dataset.iloc[:, 1]


target = []
for emotion in temp:
    if emotion == 'happy':  target.append(1)
    elif emotion == 'sad':  target.append(2)
    elif emotion == 'angry':  target.append(3)
    elif emotion == 'surprised':  target.append(4)
    else:   target.append(0)

target = numpy.array(target)

output_layer = [[0 for _ in range(5)] for _ in range(297)]
for i in range(297):
    output_layer[i][target[i]] = 1
target = numpy.array([numpy.array(y_i) for y_i in output_layer])

#x_train, x_validate, y_train, y_validate = train_test_split(x_train, target, test_size= 0.1)

model = Sequential([Dense(4048, activation= 'relu', input_shape= (43200,)), Dense(512, activation= 'relu'), Dense(64, activation= 'relu'), Dense(5, activation= 'softmax'),])
model.compile(optimizer= 'sgd', loss= 'binary_crossentropy', metrics= ['accuracy'])
model.fit(x_train, target, batch_size= 64, epochs= 150)#, validation_data= (x_validate, y_validate))

prediction = model.predict(x_test)
prediction = [numpy.argmax(vector) for vector in prediction]

print(prediction)

emotions = []
for i in prediction:
    if i == 1:  emotions.append('happy')
    elif i == 2:  emotions.append('sad')
    elif i == 3:  emotions.append('angry')
    elif i == 4:  emotions.append('surprised')
    else:   emotions.append('Unknown')

print(emotions)

f = open('temp.csv', "w")
f.write('Frame_ID,Emotion\n')
for i,emotion in enumerate(emotions):
    f.write('test'+ str(i)+ '.jpg,'+ emotion+ '\n')

f.close()