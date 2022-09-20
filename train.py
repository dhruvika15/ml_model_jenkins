from keras.models import Sequential
model = Sequential()
from keras.layers import Convolution2D
model.add(Convolution2D(
filters = 32,
kernel_size=(3,3),
input_shape=(64,64,3)))
from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2,2)))
from keras.layers import Flatten
model.add(Flatten())
from keras.layers import Dense
model.add(Dense(
units=128,
activation='relu'))
model.add(Dense(
units=64,
activation='relu'))
model.add(Dense(
units=32,
activation='relu'))
model.add(Dense(
units=1,
activation='sigmoid'))
#model.get_config()
#model.summary()
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'dogs-vs-cats/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'dogs-vs-cats/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator)
model.save("dogcat.h5")
