# 2 CNN Models for Classification Malaria-infected Cells Using Tensorflow Keras

## The Datasets
#### Context
The Datasets is provided [here](https://lhncbc.nlm.nih.gov/publication/pub9932), alongside with [this Publication](https://peerj.com/articles/4568/#).
And also can be found in [kaggle.com](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria). 
This Datasets contains 2 classes, 'Parasitized' & 'Uninfected'. Each class have 13779 images of cell in various sizes.

More information of this datasets can be found on the links that provided.


## Prefix
The process of this research is carried out by utilizing Google Collaboratory and using the Tensorflow-Keras library of python

### Preparing data
```markdown 
%cd '/content/drive/My Drive/Colab_Notebooks/'

!unzip 'cell_images.zip'
```
Extract Data from archive file. Unzip the Datasets to a folder in Drive

### Import all needed Library
```markdown
import cv2
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
import numpy as np
import datetime
import itertools
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report,f1_score, precision_score, recall_score
```
Here we are imported some important library to our script. such us Matplotlib, Opencv-python, Numpy, Keras, Tensorflow and Scikit-learn

### Load The Datasets
```markdown
parasitized = os.listdir('/content/drive/My Drive/Colab_Notebooks/cell_images/Parasitized/')
print(parasitized[:5])

uninfected = os.listdir('/content/drive/My Drive/Colab_Notebooks/cell_images/Uninfected/')
print(uninfected[:5])
```
We have 2 classes of images that separated into 2 folders, 'parasitized' and 'uninfected'

```markdown
plt.figure(figsize = (20,20))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = cv2.imread('/content/drive/My Drive/Colab_Notebooks/cell_images/Parasitized' + "/" + parasitized[i])
    plt.imshow(img)
    plt.title('PARASITIZED : 1')
    plt.tight_layout()
plt.show()

plt.figure(figsize = (20,20))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = cv2.imread('/content/drive/My Drive/Colab_Notebooks/cell_images/Uninfected' + "/" + uninfected[i])
    plt.imshow(img)
    plt.title('UNINFECTED : 0')
    plt.tight_layout()
plt.show()
```
Then, we can visualized some sample data that we use

![Alt text](https://user-images.githubusercontent.com/64162824/96108524-e10f5c00-0f07-11eb-8306-aa66707905ba.PNG?raw=true "Sample of Parasitized Cells")

![Alt text](https://user-images.githubusercontent.com/64162824/96108557-ea98c400-0f07-11eb-9326-6ffff699c96c.PNG?raw=true "Sample of Uninfected Cells")

```markdown
data=[]
labels=[]

for img in parasitized:
  try:
    image=plt.imread('/content/drive/My Drive/Colab_Notebooks/cell_images/Parasitized/'+img)
    image_resize = cv2.resize(image,(44,44))
    img_array = img_to_array(image_resize)
    data.append(img_array)
    labels.append(1)
  except:
    None

for img in uninfected:
  try:
    image=plt.imread('/content/drive/My Drive/Colab_Notebooks/cell_images/Uninfected/'+img)
    image_resize = cv2.resize(image,(44,44))
    img_array = img_to_array(image_resize)
    data.append(img_array)
    labels.append(0)
  except:
    None

plt.imshow(data[100])
plt.show()
```
Then we can process the images from those 2 class folders.
All images from both classes can be resized to 44x44 size to decrease of memory usage during training, then convert all images into array.

For images from 'Parasitized' folder, label it to '1', and images from 'Uninfected' folder label it to '0'. 

Then append the array of image to 'data' and label to 'labels' . This may take several hours to finish.

```markdown
image_data = np.array(data)
labels = np.array(labels)

idx = np.arange(image_data.shape[0])
np.random.shuffle(idx)
image_data = image_data[idx]
labels = labels[idx]
```
Convert the array images and the labels to numpy array then shuffle them randomly

```markdown
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(image_data, labels, test_size = 0.10, random_state = 10)

label_train = to_categorical(label_train)
label_test = to_categorical(label_test)

print(f'SHAPE OF TRAINING IMAGE DATA : {feature_train.shape}')
print(f'SHAPE OF TESTING IMAGE DATA : {feature_test.shape}')
print(f'SHAPE OF TRAINING LABELS : {label_train.shape}')
print(f'SHAPE OF TESTING LABELS : {label_test.shape}')
```
Then we can split the image data into Training sets and Testing sets. Training sets consist of feature_train & label_train, Testing sets consist of feature_test & label_test.
Splitting image data using 90 for training : 10 testing ratio. Then, label_train and label_test is encoded using tf.keras.utils.to_categorical.

Here the shape of feature_train, label_train, feature_test, label_test
```markdown
SHAPE OF TRAINING IMAGE DATA : (24802, 44, 44, 3)
SHAPE OF TESTING IMAGE DATA : (2756, 44, 44, 3)
SHAPE OF TRAINING LABELS : (24802, 2)
SHAPE OF TESTING LABELS : (2756, 2)
```
Training Image Data (feature_train) contain 24802 image data with shape (44,44,3)
Testing Image Data (feature_test) contain 2756 image data with shape (44,44,3)
Training Labels Data (label_train) contain 24802 labels data
Testing Labels Data (label_test) contain 2756 labels data

**THIS CODE BELOW IS OPTIONAL**
```markdown

# For save our numpy array data to h5 files

with h5py.File('feature_train.h5', 'w') as hf:
    hf.create_dataset("feature_train",  data=feature_train)

with h5py.File('feature_test.h5', 'w') as hf:
    hf.create_dataset("feature_test",  data=feature_test)

with h5py.File('label_train.h5', 'w') as hf:
    hf.create_dataset("label_train",  data=label_train)

with h5py.File('label_test.h5', 'w') as hf:
    hf.create_dataset("label_test",  data=label_test)
    
# for reload our h5 files to numpy array

with h5py.File('feature_train.h5', 'r') as hf:
    feature_train_ = hf['feature_train'][:]

with h5py.File('feature_test.h5', 'r') as hf:
    feature_test_ = hf['feature_test'][:]

with h5py.File('label_train.h5', 'r') as hf:
    label_train_ = hf['label_train'][:]

with h5py.File('label_test.h5', 'r') as hf:
    label_test_ = hf['label_test'][:]
```
We can save our training and testing data from numpy array to h5 file. 
The goal is to be more efficient in loading data if it will be used in making simple apps such as creating a Streamlit web app and reducing randomness from the results of splitting data.

### Making CNN models 
In this research we use 2 models of CNN. to simplify, lets call it 'Complex CNN' and 'Simple CNN'.

#### Complex CNN architecture
```markdown
model= Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),strides=1,input_shape = (44,44,3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32,kernel_size=(5,5),strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5,5),strides=2))
model2.add(BatchNormalization(axis=-1))
model2.add(Dropout(0.5))

model.add(Conv2D(filters=64,kernel_size=(5,5),strides=1))
model.add(Activation('relu'))
model.add(Conv2D(filters=64,kernel_size=(5,5),strides=1))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model2.add(BatchNormalization(axis=-1))
model2.add(Dropout(0.5))

model.add(Conv2D(filters=128,kernel_size=(5,5),strides=1,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256,kernel_size=(4,4),strides=1,padding='same'))
model2.add(BatchNormalization(axis=-1))
model2.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.summary()
```
This architecture consist of 3 blocks convolutional and 3 Dense layer.

First block consist of 2 conv_layer, 2 Activation Layer (ReLU) and 1 Sub-Sampling Layer (Max-pooling)

Second block consist of 2 conv_layer, 1 Activation Layer (ReLU) and 1 Sub-Sampling Layer (Max-pooling)

Third block consist of 2 conv_layer and 1 Activation Layer (ReLU)

BatchNormalization Layer and Dropout is used in the end of every layers

And have 3 Dense/Fully Connected Layer. 2 Dense layers contain 256 neurons with 'ReLU' activations, the last layer for classification contain 2 neurons with 'Softmax' acivations

Below the summary of 'Complex CNN' architecture
```markdown
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 40, 40, 32)        2432      
_________________________________________________________________
activation (Activation)      (None, 40, 40, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 36, 36, 32)        25632     
_________________________________________________________________
activation_1 (Activation)    (None, 36, 36, 32)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 16, 16, 32)        128       
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        51264     
_________________________________________________________________
activation_2 (Activation)    (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 64)          102464    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 64)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 3, 3, 64)          256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 3, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 128)         204928    
_________________________________________________________________
activation_3 (Activation)    (None, 3, 3, 128)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 3, 256)         524544    
_________________________________________________________________
batch_normalization_2 (Batch (None, 3, 3, 256)         1024      
_________________________________________________________________
dropout_2 (Dropout)          (None, 3, 3, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               590080    
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 514       
=================================================================
Total params: 1,569,058
Trainable params: 1,568,354
Non-trainable params: 704
```





















































## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/NazillyMada/Classification-of-Malaria-infected-Cells/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/NazillyMada/Classification-of-Malaria-infected-Cells/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
