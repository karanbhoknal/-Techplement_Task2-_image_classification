# Project Name = Image Recogination or object Detection 
# Image Classification Task 2

# Image Recognition using TensorFlow
In this article, we’ll create an image recognition model using TensorFlow and Keras. TensorFlow is a robust deep learning framework, and Keras is a high-level API(Application Programming Interface) that provides a modular, easy-to-use, and organized interface to solve real-life deep learning problems.

# Image Recognition:
In Image recognition, we input an image into a neural network and get a label (that belongs to a pre-defined class) for that image as an output. There can be multiple classes for the labeled image. If it belongs to a single class, then we call it recognition; if there are multiple classes, we call it classification. 

# Build input pipeline: 
Tensorflow APIs allow us to create input pipelines to generate input data and preprocess them effectively for the training process. The pipeline for an image model aggregates data from files in a distributed file system applies random perturbations to each image and merges randomly selected images into a batch for training.

# Build the model:
In this stage, we make choices about parameters and hyperparameters and make decisions about the number of layers to be used in our model. We decide on the input and output sizes of the layers, along with the activation function.
# Train the model: 
After creating a model, we must create an instance of the model and fit it with our training data. 

# Test the model:
The crucial part of this stage is to estimate: the amount of time the model takes to train and specify the length of training for a network depending on the number of epochs to train over.
Evaluate and improve the accuracy: Evaluating a model means comparing its performance against a validation dataset to analyze its performance through different metrics. The most common metric is ‘Accuracy’, calculated by dividing the amount of correctly classified images by the total number of images in our dataset.

# Step 1 : Importing TensorFlow and other libraries [on cmd]
The first step is to import the necessary libraries and modules such as pyplot, NumPy library, tensor-TensorFlow, os, and PIL.

# import libraries  command
1.pip install streamlit  OR
2.python -m pip install streamlit

# python Jupyter notebook run this code
# Importing libraries 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

# Step 2: Loading image datasets

data_train_path = 'Fruits_Vegetables/train'
data_test_path = 'Fruits_Vegetables/test'
data_val_path = 'Fruits_Vegetables/validation'

img_width = 180
img_height =180 

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False)

  data_cat['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
...
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']


 data_val = tf.keras.utils.image_dataset_from_directory(data_val_path,
                                                       image_size=(img_height,img_width),
                                                       batch_size=32,
                                                        shuffle=False,
                                                       validation_split=False)


data_test = tf.keras.utils.image_dataset_from_directory(
data_test_path,
    image_size=(img_height,img_width),
    shuffle=False,
    batch_size=32,
    validation_split=False
)

# step 3:  Visualizing the datasets: 
plt.figure(figsize=(10,10))
for image, labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_cat[labels[i]])
        plt.axis('off')

#  Step 4: Creating the model:
from tensorflow.keras.models import Sequential

data_train_model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))
                  
])

# Step 5: Compiling the model:
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


epochs_size = 25
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

# Visualizing result on training dataset:
epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'],label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['loss'],label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'],label = 'Validation Loss')
plt.title('Loss')

image = 'corn.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

model.save('Image_classify.keras')

# how to activate the virtual environment in python[activate the venv in system]

1.python -m venv ./venv

2.[.venv\Scripts\activate.ps1]


# After that create file [app.py] on vscode 
---------------------------------------------------------------
app.py
---------------------------------------------------------------
# Importing libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 


st.header('ML/AI Image Classification Model')

#C:\image_classification\Image_classify.keras
model = load_model('C:\image_classification\Image_classify.keras')
data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach','sweetcorn','sweetpotato','tomato','turnip','watermelon']

img_height = 180
img_width = 180

image =st.text_input('Enter Image name','Apple1.jpg')
#image =st.text_input('Enter Image name')
image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))


