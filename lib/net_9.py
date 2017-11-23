import numpy as np
import pickle
import keras
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#----------------------------------------------------------------------------------------------------------------------

learn_path = str(sys.argv[1])
check_path = str(sys.argv[2])

models_path = str(sys.argv[3])
weight_path = str(sys.argv[4])

rr = int(sys.argv[5])
ms = int(sys.argv[6])
epochs = int(sys.argv[7])
batch_size = int(sys.argv[8])

#----------------------------------------------------------------------------------------------------------------------

size = 2 * rr + 1
num_classes = 2
np.random.seed(ms)

#----------------------------------------------------------------------------------------------------------------------

learn_file = open(learn_path, "rb")
check_file = open(check_path, "rb")

learn_dict = pickle.load(learn_file)
check_dict = pickle.load(check_file)

learn_file.close()
check_file.close()

#----------------------------------------------------------------------------------------------------------------------

learn_X_d3 = learn_dict['X_d3']
learn_Y_d1 = learn_dict['Y_d1']

check_X_d3 = check_dict['X_d3']
check_Y_d1 = check_dict['Y_d1']

#----------------------------------------------------------------------------------------------------------------------

learn_X_d3 = learn_X_d3.reshape(learn_X_d3.shape[0], size * size)
check_X_d3 = check_X_d3.reshape(check_X_d3.shape[0], size * size)

#----------------------------------------------------------------------------------------------------------------------

input_shape = (size * size)

#----------------------------------------------------------------------------------------------------------------------

learn_Y_d1 = np_utils.to_categorical(learn_Y_d1, num_classes)
check_Y_d1 = np_utils.to_categorical(check_Y_d1, num_classes)

#----------------------------------------------------------------------------------------------------------------------

print('learn_X_d3.shape: ' + str(learn_X_d3.shape))
print('learn_Y_d1.shape: ' + str(learn_Y_d1.shape))

print('check_X_d3.shape: ' + str(check_X_d3.shape))
print('check_Y_d1.shape: ' + str(check_Y_d1.shape))

#----------------------------------------------------------------------------------------------------------------------

model = Sequential()

model.add(Dense(72, activation = 'relu', input_shape = (input_shape,)))

model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

#----------------------------------------------------------------------------------------------------------------------

models_json = model.to_json()

with open(models_path, "w") as json_file:

    json_file.write(models_json)

print("Saved model to disk")

#----------------------------------------------------------------------------------------------------------------------

optimizer = keras.optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = 1e-10, decay = 0.0)

loss = keras.losses.categorical_crossentropy

model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

#----------------------------------------------------------------------------------------------------------------------

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath = weight_path,
    verbose = 1,
    save_best_only = True,
    monitor = 'val_acc')

#----------------------------------------------------------------------------------------------------------------------

results = model.fit(
    learn_X_d3,
    learn_Y_d1,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_data = (check_X_d3, check_Y_d1),
    shuffle = True,
    callbacks = [checkpoint])