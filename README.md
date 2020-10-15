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
