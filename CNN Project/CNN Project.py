#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Code by Vy Ngo
#importing each of the libraries we'll need
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Dropout
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# dimensions of our images.
img_width, img_height = 150, 150

#the directories where our train and test data is
train_data_dir = 'training/training'
validation_data_dir = 'validation/validation'

#load batches at 32 at a time
batch_size = 32
number_train_examples = 1000
number_validation_examples = 200


# In[3]:


# used to rescale the pixel values from [0, 255] to between 0 and 1
datagen = ImageDataGenerator(rescale=1./255)

#Train and test data in respective directory
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


# In[4]:


#setting up a sequential model with 2 convolutional layers, each with 32 feature maps
#from regions that are 3x3 in the image
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Flatten()) #flattens the convolutional layer so it can go into a fully-connected layer
model.add(Dense(32)) #fully-connected layer
model.add(Dense(10,activation='softmax')) #generalized sigmoid


# In[5]:


#compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[6]:


#training model at some number of epochs
training_results = model.fit_generator(
        train_generator, #training set
        steps_per_epoch=number_train_examples // batch_size, 
        epochs=3, #number of epochs 
        validation_data=validation_generator, #testing set
        validation_steps=number_validation_examples // batch_size 
        )


# In[7]:


#using the results returned by the training history to visualize
#accuracy on the training and testing sets for each epoch
plt.plot(training_results.history['acc'])
plt.plot(training_results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#very inaccurate model


# In[8]:


model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(32, (3, 3),activation = 'relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(64, (3, 3),activation = 'relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(64,activation = 'relu'))
model2.add(Dense(10,activation='softmax'))


model2.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


# In[9]:


#train the second model
training_results2 = model2.fit_generator(
    train_generator,
    steps_per_epoch=number_train_examples // batch_size,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=number_validation_examples // batch_size)


# In[10]:


#visualize the results of the second model
plt.plot(training_results2.history['acc'])
plt.plot(training_results2.history['val_acc'])
plt.title('model accuracy w/ 3 pooling layers')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[11]:


model3 = Sequential()
model3.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(32, (3, 3),activation = 'relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(64, (3, 3),activation = 'relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Flatten())
model3.add(Dense(64,activation = 'relu'))
model3.add(Dropout(0.5))
model3.add(Dense(10,activation='softmax'))


model3.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


# In[12]:


#train the third model
training_results3 = model3.fit_generator(
    train_generator,
    steps_per_epoch=number_train_examples // batch_size,
    epochs=60,
    validation_data=validation_generator,
    validation_steps=number_validation_examples // batch_size)


# In[13]:


#visualize the results of the second model
plt.plot(training_results3.history['acc'])
plt.plot(training_results3.history['val_acc'])
plt.title('model accuracy w/ added 50% dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[14]:


#adding new layer to fix overfitting
model4 = Sequential()
model4.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Conv2D(32, (3, 3),activation = 'relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Conv2D(64, (3, 3),activation = 'relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Flatten())
model4.add(Dense(64,activation = 'relu'))
model4.add(Dropout(0.3))
model4.add(Dense(10,activation='softmax'))


model4.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


# In[15]:


#train the third model
training_results4 = model4.fit_generator(
    train_generator,
    steps_per_epoch=number_train_examples // batch_size,
    epochs=60,
    validation_data=validation_generator,
    validation_steps=number_validation_examples // batch_size)


# In[16]:


#visualize the results of the fourth model
plt.plot(training_results4.history['acc'])
plt.plot(training_results4.history['val_acc'])
plt.title('model4 accuracy w/ added layer')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[17]:


#changing strides to change input number
model5 = Sequential()
model5.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model5.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model5.add(Conv2D(32, (3, 3),activation = 'relu'))
model5.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model5.add(Conv2D(64, (3, 3),activation = 'relu'))
model5.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model5.add(Flatten())
model5.add(Dense(64,activation = 'relu'))
model5.add(Dropout(0.3))
model5.add(Dense(10,activation='softmax'))


model5.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


# In[18]:


#train the stride=2 model
training_results5 = model5.fit_generator(
    train_generator,
    steps_per_epoch=number_train_examples // batch_size,
    epochs=60,
    validation_data=validation_generator,
    validation_steps=number_validation_examples // batch_size)


# In[19]:


#visualize the results of the fifth model
plt.plot(training_results5.history['acc'])
plt.plot(training_results5.history['val_acc'])
plt.title('model4 accuracy w/ added layer and half inputs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[23]:


#changing strides to change input number
model6 = Sequential()
model6.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(img_width, img_height, 3)))
model6.add(MaxPooling2D(pool_size=(2, 2)))

model6.add(Conv2D(32, (3, 3),activation = 'relu'))
model6.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model6.add(Conv2D(64, (3, 3),activation = 'relu'))
model6.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model6.add(Flatten())
model6.add(Dense(64,activation = 'relu'))
model6.add(Dropout(0.3))
model6.add(Dense(10,activation='softmax'))


model6.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


# In[29]:


#train with early stopping
training_results6 = model6.fit_generator(
    train_generator,
    steps_per_epoch=number_train_examples // batch_size,
    epochs=60,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)],
    validation_data=validation_generator,
    validation_steps=number_validation_examples // batch_size)


# In[30]:


#visualize the results of the fifth model
plt.plot(training_results6.history['acc'])
plt.plot(training_results6.history['val_acc'])
plt.title('model4 accuracy w/ added layer and half inputs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




