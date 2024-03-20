# EXPERIMENT 03: CONVOLUTIONAL DEEP NEURAL NETWORK FOR DIGIT CLASSIFICATION
## AIM:
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.
## PROBLEM STATEMENT AND DATASET:
Problem Statement:<br/>
The task at hand involves developing a Convolutional Neural Network (CNN) that can accurately classify handwritten digits ranging from 0 to 9. This CNN should be capable of processing scanned images of handwritten digits, even those not included in the standard dataset.

Dataset:<br/>
The MNIST dataset is widely recognized as a foundational resource in both machine learning and computer vision. It consists of grayscale images measuring 28x28 pixels, each depicting a handwritten digit from 0 to 9. The dataset includes 60,000 training images and 10,000 test images, meticulously labeled for model evaluation. Grayscale representations of these images range from 0 to 255, with 0 representing black and 255 representing white. MNIST serves as a benchmark for assessing various machine learning models, particularly for digit recognition tasks. By utilizing MNIST, we aim to develop and evaluate a specialized CNN for digit classification while also testing its ability to generalize to real-world handwritten images not present in the dataset.

## NEURAL NETWORK MODEL
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/c3f13fc2-1b61-49a1-a72d-54274b33742c)

## DESIGN STEPS
### STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.
### STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.
### STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.
### STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.
### STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

## PROGRAM:
### Name: RITHIGA SRI.B
### Register Number: 212221230083
```python
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

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
print("Rithiga Sri.B 212221230083")
print("Rithiga Sri.B 212221230083")
metrics[['accuracy','val_accuracy']].plot()
print("Rithiga Sri.B 212221230083")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Rithiga Sri.B 212221230083")
print(confusion_matrix(y_test,x_test_predictions))

print("Rithiga Sri.B 212221230083")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/img.jpeg')
type(img)

img = image.load_img('/content/six.png')
plt.imshow(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print("Rithiga Sri.B 212221230083")
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img1 = image.load_img('/content/zero.jfif')
plt.imshow(img1)
img_tensor1 = tf.convert_to_tensor(np.asarray(img1))
img_28_gray1 = tf.image.resize(img_tensor1,(28,28))
img_28_gray1 = tf.image.rgb_to_grayscale(img_28_gray1)
img_28_gray_inverted1 = 255.0-img_28_gray1
img_28_gray_inverted_scaled1 = img_28_gray_inverted1.numpy()/255.0

x_single_prediction1 = np.argmax(model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),axis=1)
print("Rithiga Sri.B 212221230083")
print(x_single_prediction1)
plt.imshow(img_28_gray_inverted_scaled1.reshape(28,28),cmap='gray')
```

## OUTPUT:
### Training Data:
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/2684544a-a072-46b3-8b4a-affbf61df885)

### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/198dd92f-5a5c-4e74-89c6-cfa86bfb7b81)
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/a45cae49-ce45-42fc-b51e-af5bae6fde64)

### Classification Report:
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/cb25da16-b52f-4201-9631-68d78e09edde)

### Confusion Matrix:
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/36f5bb5f-f6cc-44df-aada-2aabea324ad9)

### New Sample Data Prediction:
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/a5482476-0885-4fc3-a83f-1448955f4af5)
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/9a169e41-294f-4aec-86ae-641f7b253b0f)
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/f142289c-1cd2-4938-a8d6-63939fce6c47)
![image](https://github.com/Rithigasri/mnist-classification/assets/93427256/ef7568af-4ece-45cc-8945-752371d32d3e)

## RESULT:
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
