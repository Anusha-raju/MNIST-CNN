# MNIST Model Training

This repository demonstrates a lightweight convolutional neural network (CNN) model for classifying MNIST digits using PyTorch. The code incorporates several steps for model development, including data loading, transformation, model architecture design, training, and evaluation. Additionally, various techniques such as batch normalization, dropout, and data augmentation are applied to improve model performance.



## Project Overview

This project focuses on designing the architecture & training a model to classify MNIST handwritten digits. Several steps are involved in building, training, and testing the model, as well as improving its accuracy using regularization techniques and data augmentation.

## Dependencies

To run this project, you need the following Python libraries:

- torch
- torchvision
- torchsummary
- tqdm
- matplotlib
- numpy

Install the dependencies with the following:

`pip install torch torchvision torchsummary tqdm matplotlib numpy`

## DATA - MNIST

The MNIST (Modified National Institute of Standards and Technology) dataset is a collection of 28x28 grayscale images of handwritten digits (0-9). The dataset contains 60,000 training images and 10,000 test images.

## Methodolgy

To progressively construct the architecture of the CNN model for predicting handwritten digit images from the MNIST dataset.



***The process is divided into 3 parts/steps:***

All the parts follow the following code execution steps:

1. Applying *Transformations* on Data

2. Splitting the data into *Train & Test Data*

3. Viewing the *Data Statistics*

4. *Visualizing Sample Images* from the MNIST Dataset

5. Building the *Model Architecture*

6. Viewing the *Model Summary*

7. *Training and testing our model* and Visualizing the accuarcy and loss curves



The *Train* data statistics :

![Train Data Statistics](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/Train%20data%20statistics.png)

Visualization of Sample Images from MNIST dataset

![Sample Images](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/Sample%20Images.png)

# Part-1

**Target**: Focusing on basic setting up the code and skeleton of the model along with keeping the model lighter.

**Model Summary**

![Model 1 Summary](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%201%20summary.png)

**Traning & Test accuracies**

![Model 1 Epoch](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%201%20epoch.png)

![Model 1 Accuracy and Loss Curves](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%201%20acc-loss%20curves.png)



**Overall Result:**

Parameters: 8,716

Best Training Accuracy: 99.04 %

Best Test Accuracy: 98.87%

**Analysis:** The model is light & no overfitting, but is not reaching the final expectation.







# Part-2

**Target**: To decrease the difference between the train and test accuarcy by adding batch normalization, regularization (dropout) , Gap(global average pooling) and also removed the last big kernel

**Model Summary**

![Model 2 Summary](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%202%20summary.png)

**Traning & Test accuracies**



![Model 2 Epoch](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%202%20epoch.png)

![Model 2 Accuracy-Loss Curves](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%202%20acc-loss%20curves.png)

**Overall Result:**

Parameters: 3,964

  Best Training Accuracy: 98.48 %

  Best Test Accuracy: 98.69%

**Analysis:** The model is more lighter but the accuracies have reduced. This is sort of predicted given that the comparision is between a 8716 parameter model to 3964 parameter model.

# Part-3

This model is trained on EC2 instance.

**Target**: 
A random rotation of 15 degrees is applied to the training sample, followed by cropping the image to 22x22 pixels. This occurs with a 10% probability. Additionally, a learning rate scheduler is set up.

**Model Summary**

![Model Summary](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%203%20summary.png)




**Traning & Test accuracies**

![Model 3 Epoch](https://raw.githubusercontent.com/Anusha-raju/MNIST-CNN/main/images/model%20%203%20epoch.png)


**Overall Result:**


  Parameters: 7,926

  Best Training Accuracy: 98.99 %

  Best Test Accuracy: 99.48%

**Analysis:**

The CNN model was successfully trained to predict digits from the MNIST dataset with fewer than 8k parameters, achieving a consistent accuracy of at least 99.4% in under 15 epochs, while minimizing the gap between training and test accuracy.



# Reference links



[Training Notebook]()

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- Pytorch for providing the deep learning framework.
