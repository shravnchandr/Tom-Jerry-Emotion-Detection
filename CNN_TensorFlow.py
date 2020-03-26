import pandas
import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

train_data = ImageDataGenerator(rescale = 1.0/255.)
test_data = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_data.flow_from_directory('Training Frames',
                                                class_mode='categorical', 
                                                target_size= (80, 45))
emotions = pandas.read_csv('Train.csv')
emotions = numpy.array(emotions.iloc[:, 1])

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(80, 45, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.01),
            loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, steps_per_epoch=100, epochs=49, verbose=1)
