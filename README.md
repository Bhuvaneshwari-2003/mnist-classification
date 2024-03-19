# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset:
```

Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.
Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).
```
## Neural Network Model

![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/09bdb20c-0750-432e-ac33-881100e8a3f6)


## DESIGN STEPS
# Step01 : Import Libraries:
tensorflow as tf (or tensorflow.keras for a higher-level API)

# Step02 : Load and Preprocess Data:
Use tf.keras.datasets.mnist.load_data() to get training and testing data. Normalize pixel values (e.g., divide by 255) for better training. Consider one-hot encoding labels for multi-class classification.

# Step03 : Define Model Architecture:
Use a sequential model (tf.keras.Sequential). Start with a Convolutional layer (e.g., Conv2D) with filters and kernel size. Add pooling layers (e.g., MaxPooling2D) for dimensionality reduction. Repeat Conv2D and MaxPooling for feature extraction (optional). Flatten the output from the convolutional layers. Add Dense layers (e.g., Dense) with neurons for classification. Use appropriate activation functions (e.g., ReLU) and output activation (e.g., softmax for 10 classes).

# Step04 : Compile the Model:
Specify optimizer (e.g., Adam), loss function (e.g., categorical_crossentropy), and metrics (e.g., accuracy).

# Step05 : Train the Model:
Use model.fit(X_train, y_train, epochs=...) to train. Provide validation data (X_test, y_test) for monitoring performance.

# Step06 : Evaluate the Model:
Use model.evaluate(X_test, y_test) to assess accuracy and other metrics.

## PROGRAM:

### Name:S.bhuvaneshwari

### Register Number:212221240010
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('img_7.png')

img = image.load_img('img_7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```

## OUTPUT


### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/355d292a-9214-47a0-add1-f71c6599cf46)

![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/740a5e6f-f4af-4ccb-bea9-872f089ebd53)

![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/ed62c815-bcf6-4678-af8f-3231fac5523d)


### Classification Report
![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/f76245ac-3468-4860-8dcd-c34a01cd385f)



### Confusion Matrix
![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/699c515a-ef98-43eb-8644-a44d2e295a77)


### New Sample Data Prediction
![image](https://github.com/Bhuvaneshwari-2003/mnist-classification/assets/94828604/7f5d0923-bd67-4682-aedd-6fe2c72443e7)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
