# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[img1]: ./writeup_images/training_set.png "Visualization"
[img2]: ./writeup_images/training_labels_hist.png "Histogram Labels"
[img3]: ./writeup_images/preprocess.png "Preprocess"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---
### 1- Dataset 

##### Summary:

- Number of training examples = 34799 (67.1%)

- Number of validation examples = 4410 (8.5%)

- Number of testing examples = 12630 (24.4%)

- Image data shape/type = (`33x32x3`) `uint8`

- Number of classes = 43

The dataset includes additional information about the original sizes of the images before being resized to 32x32, as well as the coordinates in the original images that delimit the sign. This extra information is not used at all in this project.

##### Exploratory visualization:

Random images samples from each one of the classes/labels, the sign is usually located in the center of the image with no cropping. The images have different levels of exposure, as visually, some of them, are almost completely dark. 

![alt text][img1]

Below is the distribution of the classes in the dataset, it's visible that some classes are more predominant in the dataset (up to a factor of 10)

![alt text][img2]

---
#### 2 - Model architecture

##### Preprocessing:

The images were converted to grayscale after applying [histogram equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html) to uniform exposure levels across images.

On the first row are displayed the original images, bottom is the result after preprocessing:

![alt text][img3]

##### Tensorflow model:

The implemented model is heavily based on LeNet 5, with the addition of three dropout layers to prevent overfitting. 


| Layer                  |     Description                                |
|------------------------|------------------------------------------------|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 5x5x16                    |
| Flatten                | outputs 400                                    |
| **Dropout**            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 43                                     |
| Softmax                |                                                |


##### Model Training:

The Adam-Optimizer is used with the training operation of reducing cross-entropy.

Hyperparameters:

- Batch Size : `128`
- Learning rate: `0.001`
- Dropout probability: `50%`
- Epochs: `40`


##### Solution Approach:


As it stands outs, the default parameters used for LeNet 5 are already sufficient to get  good accuracy over the training set. Only the additional dropout layers were added to further increase the accuracy. Some experimentation was done by feeding RGB images to the model instead of grayscale, but based on the accuracy levels there's no evidence that color images perform better.

The test accuracy obtained is around `94%`


#### 3 - Testing new images

##### Acquiring New Images:

A set of traffic sign photos was obtained through a google images search, and cropped more or less accordingly to the data set, but not exactly. The images have different resolutions and sizes so a resizing to 32x32 was performed using `cv2.resize`.

The stop signal image in particular is an attempt to fool the A.I. of autonpmous driving vehicles.  




The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

##### Performance on New Images:

The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

##### Model Certainty - Softmax Probabilities:

The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

#### 4 - Improvements

- Augmenting the training set might help improve model certainity. Common data augmentation techniques include rotation, translation, zoom, flips, inserting jitter, and/or color perturbation. Tensorflow v2 seems to already have embedded tools for this purpose.

- Perform error analysis to identify which image characteristics the model has a harder time to classify.   

- Visualization of layers in the neural network, as suggested in the project, to get more insight on how the model works. Also exploring [Tensorboard](https://www.tensorflow.org/tensorboard) could be useful. 

