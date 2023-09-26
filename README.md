# MNIST Multi-Layer Perceptron
This script implements a Multi-Layer Perceptron (MLP) for classifying images of handwritten digits from the MNIST dataset.

The MNIST dataset (Modified National Institute of Standards and Technology dataset) is a popular dataset of images of handwritten digits. It has been referred to as the “Hello world” of deep learning datasets, and is commonly used for the testing of image-classification networks. The MNIST dataset comprises 70,000 images (60,000 training images and 10,000 testing images), each sized at 28x28 pixels.

This code was my first attempt at implementing a neural network, and was achieved through the use of a number of online tutorials and resources.

# Network description
## Input layer
The input layer of the network contains 784 neurons corresponding to each pixel in each input 28x28 pixel image. Each one of these neurons hold a number which constitutes the greyscale value of the corresponding pixel – ranging from 0 (black) to 1 (white), after scaling.

## Hidden layers
The network contains two hidden layers, each arbitrarily comprising 16 neurons. Although largely arbitrary, this structure was chosen to keep the computational demand of the network low, as well as offer a simple starting point from which to understand the architecture – to later tweak and adjust to refine its functioning.

In actuality, the number of hidden layers in the network is important because this can govern the complexity of the representations of the input data that the network can hold, and therefore use in its learning and prediction of new, unfamiliar inputs. The number of neurons within each hidden layer is also important, because it determines how many features or representations the network can learn from the input data. If there are too few neurons, the network may not be able to capture all the relevant patterns in the data. On the other hand, if there are too many neurons, the network may start to learn noise or irrelevant patterns in the data, which can lead to overfitting – where the network relies too heavily on certain neurons in the hidden layer that are not actually useful for the task at hand, leading to a situation where many of the hidden layer neurons are active for a given input, even if only a few of them are actually needed to make an accurate prediction, making the network computationally inefficient and less capable of interpreting inputs.

## Output layer
The output layer contains ten neurons, each corresponding to the possible digits that the image being read might show (digits 0 to 9). Activation in these neurons represents the ‘confidence’ of the model in that neuron/digit being the one in the input image. When the signals have travelled from the input to this output layer, the neuron/digit with the highest value is chosen as the prediction (the digit that the model predicts that it is reading).

# Implementation
The script implements this network through the following steps:

## 1) Import necessary libraries.
This script used the Python packages TensorFlow and Keras to access the MNIST data, build the network and train/test the model. MatPlotLib was also imported at the beginning of the script, simply to inspect examples of the handwritten images contained within the dataset.


## 2) Load MNIST dataset.
The tensorflow.keras.datasets module provides a number of datasets for use in training and testing machine learning models. The MNIST dataset was imported from this module.

## 3) Scale pixel values to range 0:1.
The images contained within the MNIST dataset are made up from pixels of standard values, ranging from 0 (black) to 255 (white). The script began by scaling/normalising the pixel values of all images to within the range 0:1.

This is an important preprocessing step in the implementation of such image classification models, as unscaled pixel values with a large dynamic range can cause issues in the activations transmitted throughout the network. Specifically, when an image is read, the activation of the neurons in the first hidden layer is calculated by multiplying the pixel values, represented as the activation level at the neurons of the input layer, with the weights and biases assigned to the channels connecting the input and hidden layer nodes. If the pixel values fed to this convolution are more extremely distributed, this can lead to disproportionately smaller/greater activation at neurons weighted away from/towards these value representations, causing the network to focus disproportionately more on those features weighted more heavily and therefore potentially miss information obtained from less extreme input features. This can also lead to slower convergence to the best possible prediction, since the weights that contribute to the activations through the network are updated through the calculation of the gradients of the loss function (explained further below). When the activations of the neurons in the network are more extreme, the computed gradients are similarly affected, which can lead to smaller or larger steps towards minimising the inaccuracy of the model.

The pixel values of the MNIST images were therefore scaled in this script, in order to control for these potential issues and ensure prediction accuracy and efficient reaching of convergence in the network.

## 4) Reshape input data into 784x1 vectors
The reshape function of the TensorFlow package was used to reshape the input data to make it readable by the to-be-built model as the input layer of the network.

The training and test image data automatically take the form of three-dimensional NumPy arrays with the shape: number of images, image height, image width, when loaded into the script. The training image data (of the shape (60,000, 28, 28)) and the test image data (of the shape (10,000, 28, 28)) were therefore reshaped in order to form the input layer of the network – which, in this case, comprises one neuron for each pixel in the input images.

## 5) Define neural network architecture
The neural network architecture was then defined using the models module of the Keras API. A sequential model was chosen, to create a network comprising a linear series of layers with fully connecting neurons. This architecture was largely inspired by the example neural network outlined in a YouTube video by the channel 3Blue1Brown.

As previously described, the input layer of the network contains 784 neurons corresponding to each pixel in each input 28x28 pixel image. Each one of these neurons hold a number which constitutes the greyscale value of the corresponding pixel – ranging from 0 (black) to 1 (white), after scaling.

The network also contains two hidden layers, each comprising 16 neurons. Although largely arbitrary, this structure was chosen to keep the computational demand of the network low, as well as offer a simple starting point from which to understand the architecture – to later tweak and adjust to refine its functioning.
As can be seen, the script uses the Keras layers module to define the hidden layers, specifying 16 neurons for each and the Rectified Linear Unit (ReLU) function to be used to determine the activations of the neurons in each layer (taking the maximum out of either zero or the sum of the weighted and biased inputs to the neurons). These were created using the layers.Dense function call, to define the layers as containing neurons fully-connected to the nodes of the other layers. In the first hidden layer, the shape of the input layer was also specified, to enable the predefined input layer to be successfully read as input to this layer.

The final output layer then contains ten neurons, each corresponding to the possible digits that the image being read might show (digits 0 to 9). Activation in these neurons represents the ‘confidence’ of the model in that neuron/digit being the one in the input image. When the signals have travelled from the input to this output layer, the neuron with the highest value is chosen as the prediction (the digit that the model predicts it has been given).
This layer was also defined as a dense (fully-connected) layer. However, the level of activation of the neurons at this layer was set to be determined by the SoftMax function. This function is useful, and was deemed preferable to the ReLU function at this stage, as it computes the activations of the final output neurons as a probability distribution across the possible prediction classes, ensuring that their sum is equal to 1 – giving a well-defined and bounded output to govern the network’s prediction.

## 6) Configure learning with MSE loss function, gradient descent optimisation, and accuracy as the metric
The script then uses the model.Compile() method from the Keras API to configure and compile the settings for the network’s learning. In this case, the network was implemented to use the mean squared error (MSE) as the loss function, stochastic gradient descent (SGD) as the optimisation algorithm, and accuracy as the metric for goodness of prediction.

Loss function:

In order to ‘teach’ or ‘train’ the network, we implement a loss/cost function. The cost function in a neural network measures the difference between the predicted output of the network and the true output, given a specific input. This is used to guide the network to minimise this difference, during training.

The implementation of the cost function depends on the specific problem at hand. In image classification, one method is to calculate the mean squared error (MSE) between the predicted output and the true output produced by the network.
-	For example, the predicted output for an input image representing a handwritten number 3, would be, ideally, zero output activations for all of the nine neurons that do not correspond to the number 3, and activation for that neuron alone.

The MSE is small when the network confidently classifies the image correctly, and large when it doesn’t appear to know what it’s doing.

During the training process, the cost function is evaluated for each input in the training set, and the average cost over a certain batch of the training inputs is typically used as a measure of the network's performance. This is because the goal of the network is to learn to make accurate predictions on many the input examples, not just one or a few of them.
For brevity, the batch size was not specified in this script. However, this can be implemented in the model.fit() function call, in the subsequent section of the code.

Optimisation:

Once the average cost is calculated, the network is trained to minimise its (the MSE) using an optimisation algorithm, such as stochastic gradient descent (SGD). Gradient descent is used to estimate the input parameter value required to find the local minimum of a function. In our case, we are looking to find the minimum of the cost function. This optimisation algorithm therefore tells the network to seek the smallest error/deviation from the predicted output (achieve the highest accuracy), by updating the weights and biases during training. This algorithm works by iteratively updating the weights and biases of the network in the direction of the negative gradient of the cost function. The negative gradient indicates the direction of steepest descent (i.e. the direction towards the bottom of the function slope), which leads to a decrease in the cost function.

Metric:

By specifying the metric, to be used in the prediction evaluation, to ‘accuracy’, the network was set up to compute the proportion of correct predictions to the total number of predictions.
I.e. accuracy = (number of correct predictions) / (total number of predictions).

Whether or not the network’s prediction is correct is determined by the later model.fit() function call, which compares the prediction with the training data labels (an array containing the actual numerical digits that each training image corresponds to), as a one-hot encoded vector (converted using the to_categorical() function).

## 7) Train model on training data
As mentioned, the Keras model.fit() function was used to train the network with the training data. Implementationally, the training data images are passed as the first argument, followed by the training data labels (converted to a one-hot encoded vector, for use in evaluating the network predictions), followed by the number of epochs through which the network will be trained on the entire training dataset.

Early stopping was used to determine an appropriate number of epochs to use, whereby the number of epochs was increased over multiple runs of the script until the accuracy of the model reached a maximal level. The epoch number producing this level was then selected.

Specifically, this section of the code ultimately implements the fundamental following steps, to train the network:
1.	The network is initialized with random weights and biases.

2.	During each iteration of training, the network makes a forward pass to compute the predicted class probabilities for the input data.

3.	The predicted class probabilities are compared to the true class labels using the MSE loss function.

4.	The gradients of the loss function with respect to the weights and biases of the network are computed using backpropagation:

  During the backward pass, the gradients of the loss function with respect to the weights and biases of the network are computed using the chain rule of calculus. The gradients are computed by working backwards through the network, starting from the output layer and moving towards the input layer. This involves computing the gradient of the loss function with respect to the output of each neuron in the network, and then using the chain rule of calculus to compute the gradient of the loss function with respect to the input to each neuron in the network.

  The training data is divided into smaller batches, and the gradients are computed and averaged over each batch. This allows the model to update its weights more frequently than if it were to wait until all training data is processed in an epoch, and can help the model converge faster and improve performance. The batch size determines the number of training examples in each batch, and it is a hyperparameter that can be tuned to achieve the desired trade-off between convergence speed and generalization performance.
Weight Update: The gradients are used to update the weights and biases of the network using the SGD optimization algorithm. The weights are updated by subtracting a fraction of the gradient from the current weights, scaled by the learning rate used by the SGD.

5.	The weights and biases of the network are updated using the computed gradients and the SGD optimisation algorithm.

6.	Steps 2-5 are repeated for the specified number of epochs.


## 8)	Evaluate network on test data
Using the evaluate() function of the Keras API model module – passing the test image data and the corresponding labels as a one-hot encoded vector, to evaluate its prediction. 

Following the ‘learning’ from the prior training, the network at this stage has optimised its parameters (the weights and biases that determine the activation configuration of the different layer neurons) to be more effective at accurately classifying the input images of handwritten digits.


## 9)	Results
Running this script shows that this simple neural network is able to achieve ~90% accuracy in its predictions of numerical values from input images of handwritten digits.



