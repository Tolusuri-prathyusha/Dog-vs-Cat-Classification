## Dog vs Cat Classification
# Introduction
This project aims to classify images of dogs and cats using machine learning techniques. The task involves training a model to distinguish between images containing dogs and images containing cats.

# Dataset
The dataset used for this project is the Dog vs Cat dataset. This dataset contains thousands of images of dogs and cats. Each image is labeled with either "dog" or "cat".

# Model Architecture
For this classification task, a Convolutional Neural Network (CNN) architecture is used. CNNs are well-suited for image classification tasks because they can effectively capture spatial hierarchies in images.

The model architecture consists of several convolutional layers followed by max-pooling layers to extract features from the input images. The final layers are fully connected layers with dropout regularization to prevent overfitting. The output layer has two units, one for each class (dog and cat), with softmax activation to produce class probabilities.

# Training
The model is trained using the training set of the Dog vs Cat dataset. During training, the model learns to minimize a categorical cross-entropy loss function by adjusting its weights and biases through backpropagation and gradient descent optimization.

# Evaluation
The performance of the model is evaluated using the test set of the Dog vs Cat dataset. Metrics such as accuracy, precision, recall, and F1-score are computed to assess the model's performance in distinguishing between dogs and cats.

[View on GitHub](https://github.com/Tolusuri-prathyusha/Dog-vs-Cat-Classification)

