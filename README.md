# MRI Image Classification for Dementia Stages

## Project Overview

This repository contains machine learning projects developed for the course CS 4641: Machine Learning at the College of Computing, Georgia Institute of Technology. The projects are focused on using MRI images to classify the extent of dementia in patients, developed by Jessica Hernandez & John Fortner under the guidance of Dr. Gerandy Britito.

### Introduction

With advancements in MRI technology, understanding brain functionality and associated damages has improved. This project aims to classify MRI images into four dementia categories using machine learning techniques. The categories are NotDemented, VeryMildDemented, MildDemented, and ModerateDemented. The classifier predicts the extent of dementia in patients, aiding in the early detection and treatment of Alzheimer’s disease.

### Data Source

The MRI images used in this project are sourced from the "Alzheimer's Dataset (4 class of Images)" on Kaggle, uploaded by Sarvesh Dubey. The dataset includes 6400 images, each labeled with one of the four dementia stages.

## Technical Documentation

### Data Format

- **Image Dimensions:** 176 x 208 x 3 channels
- **Preprocessing:** Images are transformed to grayscale to reduce computational load during training.

### Model Architecture

- **Convolutional Neural Network (CNN)** implemented using PyTorch.
- **Layers:**
  - Convolutional layers with ReLU activation and max pooling.
  - Dropout layers to prevent overfitting.
  - Fully connected layers with linear transformation and ReLU activation.

### Optimization and Loss

- **Optimizer:** Adamax, selected for its efficacy in handling image data.
- **Loss Function:** Cross Entropy Loss, suitable for multi-class classification.

## Findings and Analysis

The final model achieved a training accuracy of 100% and a testing accuracy of 65%, indicating some overfitting despite attempts to mitigate it through dropout layers and optimizer adjustments. Given more resources, the accuracy and stability of the model could potentially be improved by exploring more complex architectures and increasing the training duration.

## Future Work

Further research could explore advanced neural network models and employ more extensive computational resources to enhance the accuracy and efficacy of dementia stage classification.

## References

- Srivastava, N., et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15, 1929-1958.
- Kingma, D. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.
- Sultana, F., et al. (2018). Advancements in Image Classification using Convolutional Neural Network. 2018 Fourth International Conference on Research in Computational Intelligence and Communication Networks.

## Contact

- **Jessica Hernandez** - [Email](mailto:jessica.hernandez@example.com)
- **John Fortner** - [Email](mailto:john.fortner@example.com)
