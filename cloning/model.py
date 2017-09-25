
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Merge, ELU
from keras.layers import Conv2D, Activation, Convolution2D, MaxPool2D, MaxPooling2D, Cropping2D, GlobalAvgPool2D
from keras.optimizers import adam, RMSprop, SGD
import matplotlib.pyplot as plt




# In[2]:

ROWS = 80
COLS = 160
CHANNELS = 3


# In[3]:

def get_model():
    """Define hyperparameters and compile model"""
    
    lr = 0.0001
    weight_kernel_initializer='glorot_normal'
    opt = RMSprop(lr, decay=1e-5)
    loss = 'mean_squared_error'

    model = Sequential()
    
    model.add(BatchNormalization(axis=1, input_shape=(ROWS, COLS, CHANNELS)))
    model.add(Convolution2D(8, (3, 3), kernel_initializer=weight_kernel_initializer, padding='valid', activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer=weight_kernel_initializer, padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), kernel_initializer=weight_kernel_initializer, padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), kernel_initializer=weight_kernel_initializer, padding='valid',  activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=weight_kernel_initializer))

    model.add(Dense(32, activation='relu', kernel_initializer=weight_kernel_initializer))
    
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer=weight_kernel_initializer, activation='linear'))

    model.compile(optimizer=opt, loss=loss)

    return model

    
model = get_model()
model.summary()


# In[4]:

image_data = []
measurement_data = []
for num in range(0,4): # using images from several training runs.
    print(num)
    path_df = pd.read_csv('training/driving_log{}.csv'.format(num), header=None)
    path_list = list(path_df.iloc[:,0])
    measurements = list(path_df.iloc[:,3])
    for i, path in enumerate(path_list):
        filename = path.split('/')[-1]
        current_path = 'training/IMG{}/'.format(num) + filename
        img = cv2.resize(
              cv2.imread(current_path),(160, 80))#,
        image_data.append(img)
        measurement_data.append(measurements[i])
        img_flipped = cv2.flip(img, 1)
        image_data.append(img_flipped)
        measurement_data.append(-measurements[i])
        
x = np.array(image_data)
y = np.asarray(measurement_data)


# In[ ]:

print('The Model will be trained using {} images'.format(x.shape[0]))


# In[5]:

model.fit(x, y, validation_split=0.2, epochs=5, shuffle=True, batch_size=20)


# In[6]:

model.save('model.h5')


# ## Data generator 

# In[ ]:

#import os
#import csv
# for num in range(1,3):
#     path_df = pd.read_csv('training/driving_log{}.csv'.format(num), header=None)
#samples = []
#with open('driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)


# In[ ]:

#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle


# In[ ]:

#train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[ ]:

#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#   while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]

#            images = []
#            angles = []
#            for batch_sample in batch_samples:
#                name = 'IMG/'+batch_sample[0].split('/')[-1]
#                image = cv2.resize(cv2.imread(name), (160, 80))
#                angle = float(batch_sample[3])
#                images.append(image)
#                angles.append(angle)
#                image_flipped = cv2.flip(image, 1)
#                images.append(image_flipped)
#                angles.append(-angle)
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)

# ch, row, col = 3, 80, 320  # Trimmed image format


# In[ ]:

#model.fit_generator(train_generator, steps_per_epoch=512, 
#  epochs=3, validation_steps=1)


# ## Visualising the model architecture

# In[ ]:

#from IPython.display import Image, display, SVG
#from keras.utils.vis_utils import pydot, model_to_dot

# Show the model in ipython notebook
#figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#display(figure)


# In[6]:

#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model.fig')

