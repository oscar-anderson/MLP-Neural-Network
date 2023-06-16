# Multi-Layer Perceptron for MNIST digit image classification.

# Import necessary libraries.
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# (Uncomment to view a random MNIST handwritten digit training image):
# import matplotlib.pyplot as plt
# import random
# plt.imshow(train_images[random.randint(0, 59000)])
# plt.show()

# Scale pixel values to range 0:1.
train_images = train_images / 255
test_images = test_images / 255

# Reshape input data into 784 x 1 vector, to form input layer.
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

# Define neural network architecture.
# Create sequential model (sequence of network layers).
model = tf.keras.models.Sequential([
    
    # Create fully-connected (Dense) first hidden layer, arbitrarily comprising 16 neurons.
    # Activation at this layer's neurons is determined using ReLU function.
    # Specify shape of previously-defined input layer as 784 x 1, using one-element tuple, for first layer.
    tf.keras.layers.Dense(16, activation='relu', input_shape=(784, )),
    
    # Create fully-connected (Dense) second hidden layer, arbitrarily comprising 16 neurons.
    # Activation at this layer's neurons is determined using ReLU function.
    # Shape of input matrix from previous layer is automatically inferred by Keras API.
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Create output layer, containing one neuron for each possible digit prediction.
    # Activation at this layer's neruons is determined using softmax function.
    # Shape of input matrix from previous layer is automatically inferred by Keras API.
    tf.keras.layers.Dense(10, activation='softmax')

])

# Configure learning with MSE loss function, gradient descent optimisation, and accuracy as metric.
model.compile(loss='mse', optimizer='sgd', metrics='accuracy')

# Train model on training data.
print('Training in progress...')
model.fit(train_images, tf.keras.utils.to_categorical(train_labels), epochs=40)
        # 1) Pass training data.
        # 2) Pass training data labels as 2D one-hot vector, to enable prediction evaluation and softmax probability distribution output.
        # 3) Pass number of epochs (iterations over training data). This can be adjusted to achieve highest accuracy.

# Evaluate model on test data.
print('Testing in progress...')
test_loss, test_acc = model.evaluate(test_images, tf.keras.utils.to_categorical(test_labels))

# Display network prediction accuracy, following testing.
print('Test complete.')
print("Test accuracy:", test_acc)

