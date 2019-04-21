import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
from keras_tqdm import TQDMNotebookCallback

#Loading the hotdog dataset
batch_size = 10
epochs = 3
# Re-scaled dimensions of images
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

from keras.applications.mobilenet import MobileNet, preprocess_input
mobilenet_base = MobileNet(weights='imagenet', include_top=False)

def mymodel():
    
    #Simple model from: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d taken and edited
   
    model = Sequential()
    model.add(mobilenet_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    for layer in mobilenet_base.layers:
        layer.trainable = False
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

# Test function
mymodel().summary()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data_dir = '/devika/rotman/hotdog/train'
test_data_dir = '/devika/rotman/hotdog/test'

    
# Data parameters 
num_train_samples = 498
num_test_samples = 500

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

#Loading data on the fly
def evaluate_model(runs=5):
    
    scores = [] 
    for i in range(runs):
        print('Executing run %d' % (i+1))
        model = mymodel()
        model.fit_generator(train_generator,
                            callbacks=[TQDMNotebookCallback()],
                            steps_per_epoch=num_train_samples // batch_size,
                            epochs=epochs, verbose=0)
        print(' * Evaluating model on test set')
        scores.append(model.evaluate_generator(test_generator, 
                                               steps=num_test_samples // batch_size,
                                               verbose=0))
        print(' * Test set Loss: %.4f, Accuracy: %.4f' % (scores[-1][0], scores[-1][1]))
        
    accuracies = [score[1] for score in scores]     
    return np.mean(accuracies), np.std(accuracies)
        
mean_accuracy, std_accuracy = evaluate_model(runs=5)

#Mean test set accuracy over 5 runs
print('Mean test set accuracy over 5 runs: %.4f +/- %.4f' % (mean_accuracy, std_accuracy))

