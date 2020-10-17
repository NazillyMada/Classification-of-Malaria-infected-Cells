###### Github Pages : https://nazillymada.github.io/Classification-of-Malaria-infected-Cells/

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

#### 'Complex CNN' architecture
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

opt = SGD(lr=0.001, momentum=0.9)

model.summary()
```
This architecture consist of 3 blocks convolutional and 3 Dense layer.

First block consist of 2 conv_layer, 2 Activation Layer (ReLU) and 1 Sub-Sampling Layer (Max-pooling)

Second block consist of 2 conv_layer, 1 Activation Layer (ReLU) and 1 Sub-Sampling Layer (Max-pooling)

Third block consist of 2 conv_layer and 1 Activation Layer (ReLU)

BatchNormalization Layer and Dropout is used in the end of every layers

And have 3 Dense/Fully Connected Layer. 2 Dense layers contain 256 neurons with 'ReLU' activations, the last layer for classification contain 2 neurons with 'Softmax' acivations

then, for Optimizer we use SGD with learing rate = 0.001 and momentum = 0.9

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
##### Train the 'Complex CNN'
```markdown
filepath='weights.h5'
my_callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience =10),
                tf.keras.callbacks.ModelCheckpoint(filepath , monitor='val_accuracy',verbose=1,save_best_only=True),
]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
my_callbacks.append(TensorBoard(logdir, histogram_freq=1))

print(my_callbacks)
```
Before we train the 'Complex CNN' models, we setting up some callbacks. Such us Checkpoint, EarlyStopping, and Tensorboard.

Checkpoint used to save the model in several epoch with increased val_accuracy

EarlyStopping used to stop the training if there is no improvement of val_accuracy from 10 epoch in a row

TensorBoard is used to see the graphics of loss,accuracy,val_accuracy, val_loss during training process

```markdown
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(feature_train,label_train, validation_data=(feature_test,label_test), callbacks = my_callbacks, epochs=100,verbose=1)
```
Then we can compile the model and fit for training.

```markdown
# 3 first epoch
Epoch 1/100
  1/776 [..............................] - ETA: 10s - loss: 0.8452 - accuracy: 0.6250
776/776 [==============================] - ETA: 0s - loss: 0.6181 - accuracy: 0.6620
Epoch 00001: val_accuracy improved from -inf to 0.67925, saving model to weights3.h5
776/776 [==============================] - 173s 223ms/step - loss: 0.6181 - accuracy: 0.6620 - val_loss: 0.5952 - val_accuracy: 0.6792
Epoch 2/100
776/776 [==============================] - ETA: 0s - loss: 0.5365 - accuracy: 0.7313
Epoch 00002: val_accuracy improved from 0.67925 to 0.73948, saving model to weights3.h5
776/776 [==============================] - 163s 210ms/step - loss: 0.5365 - accuracy: 0.7313 - val_loss: 0.5216 - val_accuracy: 0.7395
Epoch 3/100
775/776 [============================>.] - ETA: 0s - loss: 0.4542 - accuracy: 0.7843
Epoch 00003: val_accuracy did not improve from 0.73948
776/776 [==============================] - 161s 208ms/step - loss: 0.4542 - accuracy: 0.7843 - val_loss: 1.0927 - val_accuracy: 0.5112

.......
# Best Epoch
Epoch 18/100
775/776 [============================>.] - ETA: 0s - loss: 0.1188 - accuracy: 0.9571
Epoch 00018: val_accuracy improved from 0.95864 to 0.95936, saving model to weights3.h5
776/776 [==============================] - 154s 199ms/step - loss: 0.1188 - accuracy: 0.9571 - val_loss: 0.1293 - val_accuracy: 0.9594

# 3 Last Epoch
Epoch 26/100
775/776 [============================>.] - ETA: 0s - loss: 0.0914 - accuracy: 0.9661
Epoch 00026: val_accuracy did not improve from 0.95936
776/776 [==============================] - 155s 199ms/step - loss: 0.0914 - accuracy: 0.9661 - val_loss: 0.1499 - val_accuracy: 0.9539
Epoch 27/100
775/776 [============================>.] - ETA: 0s - loss: 0.0818 - accuracy: 0.9704
Epoch 00027: val_accuracy did not improve from 0.95936
776/776 [==============================] - 154s 199ms/step - loss: 0.0818 - accuracy: 0.9704 - val_loss: 0.1707 - val_accuracy: 0.9546
Epoch 28/100
775/776 [============================>.] - ETA: 0s - loss: 0.0764 - accuracy: 0.9725
Epoch 00028: val_accuracy did not improve from 0.95936
776/776 [==============================] - 157s 202ms/step - loss: 0.0764 - accuracy: 0.9725 - val_loss: 0.1290 - val_accuracy: 0.9565
```

**TensorBoard**

![Alt text](https://user-images.githubusercontent.com/64162824/96118966-7ade0580-0f16-11eb-927e-639c167790c3.PNG?raw=true "Epoch Accuracy")

![Alt text](https://user-images.githubusercontent.com/64162824/96118977-7f0a2300-0f16-11eb-8547-96a3d0a51641.PNG?raw=true "Epoch Loss")

##### 'Complex CNN' Model Evaluation

After training, evaluate our model with testing sets to figure out how well our model make predictions

```markdown
model_ = load_model('/content/drive/My Drive/Colab_Notebooks/weights.h5')
```
Before we perform evaluation, load the best model & weights from training

```markdown
# Perform evaluation
loss, accuracy = model_.evaluate(feature_test,label_test)
f1score = f1_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")
precision = precision_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")
recall = recall_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")

# Result
Loss : 0.0788116306066513 , Accuracy : 0.9735123515129089
F1 Score : 0.9735122495379178
Precision : 0.9735818698435521
Recall : 0.9735556819498425
```

**Confusion Matrix**

```markdown
def plot_confusion_matrix(cm,classes,normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    print("normalized confusion matrix")
  else:
    print("Confusion Matrix")
  
  print(cm)
  thresh = cm.max()/2.
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j], horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
  

  plt.tight_layout()
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')

cm = confusion_matrix(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1))
cm_plot_labels = ['Uninfected','Parasitized']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')
```

![Alt text](https://user-images.githubusercontent.com/64162824/96121318-1d4bb800-0f1a-11eb-985f-669072cd0925.PNG?raw=true "Confusion Matrix")

We also using Confusion Matrix to see our prediction from each class

If we see the prediction result of 'Complex CNN' model with the testing set, we can say that is pretty good !


#### 'Simple CNN' Architecture

```markdown
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape = (44,44,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

model.add(Flatten())

#fully-connected layer
model.add(Dense(units=64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(units=128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(units=32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(units=2, activation='softmax'))

opt = SGD(lr=0.001, momentum=0.9)

model.summary()
```
'Simple CNN' Model id made of 1 blocks Convolutional layers and 4 Dense Layer

'Simple CNN' consist of 2 Conv layer, 2 Sub-Sampling Layer (Max-Pooling) and BatchNormalization + Dropout Layer in the end

And have 4 Dense/ Fully-Connected Layer that consist of Dense_1 with 64 neurons, Dense_2 with 128 neurons, Dense_3 with 32 neurons, all using 'ReLU' activation.

and Dense_4 for classification using 2 neurons with 'Softmaxt' activation

then, for Optimizer we use SGD with learing rate = 0.001 and momentum = 0.9

```markdown
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_11 (Conv2D)           (None, 42, 42, 32)        896       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 21, 21, 32)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 19, 19, 64)        18496     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 9, 9, 64)          0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 9, 9, 64)          256       
_________________________________________________________________
dropout_5 (Dropout)          (None, 9, 9, 64)          0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 5184)              0         
_________________________________________________________________
dense_12 (Dense)             (None, 64)                331840    
_________________________________________________________________
dense_13 (Dense)             (None, 128)               8320      
_________________________________________________________________
dense_14 (Dense)             (None, 32)                4128      
_________________________________________________________________
dense_15 (Dense)             (None, 2)                 66        
=================================================================
Total params: 364,002
Trainable params: 363,874
Non-trainable params: 128
```
#### Train 'Simple CNN'

```markdown
filepath='weights3.h5'
my_callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience =10),
                tf.keras.callbacks.ModelCheckpoint(filepath , monitor='val_accuracy',verbose=1,save_best_only=True),
]

logdir = os.path.join("logs1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
my_callbacks.append(TensorBoard(logdir, histogram_freq=1))

print(my_callbacks)
```
Before we train the 'Complex CNN' models, we setting up some callbacks. Such us Checkpoint, EarlyStopping, and Tensorboard.

Checkpoint used to save the model in several epoch with increased val_accuracy

EarlyStopping used to stop the training if there is no improvement of val_accuracy from 10 epoch in a row

TensorBoard is used to see the graphics of loss,accuracy,val_accuracy, val_loss during training process

```markdown
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(feature_train,label_train, validation_data=(feature_test,label_test), callbacks = my_callbacks, epochs=100,verbose=1)
```
Then we can compile the model and fit for training.

```markdown
# 3 first epoch
Epoch 1/100
775/776 [============================>.] - ETA: 0s - loss: 0.6151 - accuracy: 0.6667
Epoch 00001: val_accuracy improved from -inf to 0.72714, saving model to weights3.h5
776/776 [==============================] - 68s 88ms/step - loss: 0.6151 - accuracy: 0.6668 - val_loss: 0.5474 - val_accuracy: 0.7271
Epoch 2/100
775/776 [============================>.] - ETA: 0s - loss: 0.5167 - accuracy: 0.7471
Epoch 00002: val_accuracy did not improve from 0.72714
776/776 [==============================] - 73s 95ms/step - loss: 0.5168 - accuracy: 0.7471 - val_loss: 0.7268 - val_accuracy: 0.6549
Epoch 3/100
775/776 [============================>.] - ETA: 0s - loss: 0.4029 - accuracy: 0.8145
Epoch 00003: val_accuracy did not improve from 0.72714
776/776 [==============================] - 65s 84ms/step - loss: 0.4029 - accuracy: 0.8145 - val_loss: 1.0150 - val_accuracy: 0.5334


......

# Best Epoch
Epoch 25/100
775/776 [============================>.] - ETA: 0s - loss: 0.0872 - accuracy: 0.9680
Epoch 00025: val_accuracy improved from 0.95718 to 0.96154, saving model to weights3.h5
776/776 [==============================] - 64s 82ms/step - loss: 0.0872 - accuracy: 0.9680 - val_loss: 0.1220 - val_accuracy: 0.9615


# 3 last epoch
Epoch 33/100
775/776 [============================>.] - ETA: 0s - loss: 0.0630 - accuracy: 0.9771
Epoch 00033: val_accuracy did not improve from 0.96154
776/776 [==============================] - 65s 83ms/step - loss: 0.0629 - accuracy: 0.9771 - val_loss: 0.1467 - val_accuracy: 0.9550
Epoch 34/100
775/776 [============================>.] - ETA: 0s - loss: 0.0595 - accuracy: 0.9779
Epoch 00034: val_accuracy did not improve from 0.96154
776/776 [==============================] - 64s 83ms/step - loss: 0.0595 - accuracy: 0.9779 - val_loss: 0.1381 - val_accuracy: 0.9608
Epoch 35/100
775/776 [============================>.] - ETA: 0s - loss: 0.0562 - accuracy: 0.9797
Epoch 00035: val_accuracy did not improve from 0.96154
776/776 [==============================] - 65s 84ms/step - loss: 0.0562 - accuracy: 0.9797 - val_loss: 0.1419 - val_accuracy: 0.9608
```
**TensorBoard**

![Alt text](https://user-images.githubusercontent.com/64162824/96126442-54bc6380-0f1e-11eb-961d-22d850dfe7db.PNG?raw=true "Epoch Accuracy")

![Alt text](https://user-images.githubusercontent.com/64162824/96126496-584fea80-0f1e-11eb-86af-3a6b457882b1.PNG?raw=true "Epoch Loss")

##### 'Simple CNN' Model Evaluation

After training, evaluate our model with testing sets to figure out how well our model make predictions

```markdown
model_ = load_model('/content/drive/My Drive/Colab_Notebooks/weights3.h5')
```
Before we perform evaluation, load the best model & weights from training

```markdown
# Perform evaluation
loss, accuracy = model_.evaluate(feature_test,label_test)
f1score = f1_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")
precision = precision_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")
recall = recall_score(np.argmax(label_test, axis = 1), np.argmax(predictions, axis = 1), average="macro")

# Result
Loss : 0.055258918553590775 , Accuracy : 0.9807692170143127
F1 Score : 0.9807692079825402
Precision : 0.9807860931222381
Recall : 0.9807959680222453
```
**Confusion Matrix**

![Alt text](https://user-images.githubusercontent.com/64162824/96128062-0cea0c00-0f1f-11eb-8b71-31ee31f2115d.PNG?raw=true "Confusion Matrix")

We also using Confusion Matrix to see our prediction from each class

If we see the prediction result of 'Simple CNN' model with the testing set, we can say that is pretty good ! and even better than the more complex one
