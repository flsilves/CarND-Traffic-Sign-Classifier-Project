# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[img1]: ./writeup_images/training_set.png "Visualization"
[img2]: ./writeup_images/training_labels_hist.png "Histogram Labels"
[img3]: ./writeup_images/preprocess.png "Preprocess"
[img4]: ./writeup_images/training.png "Training"
[img5]: ./writeup_images/web_images.png "Web Images"
[img6]: ./writeup_images/softmax_priority.png "Priority"
[img7]: ./writeup_images/softmax_stop_simple.png "Stop1"
[img8]: ./writeup_images/softmax_roundabout.png "Roundabout"
[img9]: ./writeup_images/softmax_stop_hard.png "Stop2"
[img10]: ./writeup_images/softmax_children.png "Children"
[img11]: ./writeup_images/softmax_60.png "60"

---
### 1- Dataset 

##### Summary:

- Number of training examples = 34799 (67.1%)

- Number of validation examples = 4410 (8.5%)

- Number of testing examples = 12630 (24.4%)

- Image data shape/type = (`33x32x3`) `uint8`

- Number of classes = 43

The dataset includes additional information about the original sizes of the images before being resized to 32x32, as well as the coordinates in the original images that delimit the sign. This extra information is not used in this project.

##### Exploratory visualization:

Random images samples from each one of the classes/labels, the sign is usually located in the center of the image with no cropping. The images have different levels of exposure, some of them are almost completely dark. 

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
- Epochs: `30`

![alt text][img4]

##### Solution Approach:

The default parameters used for LeNet 5 are already sufficient to get good accuracy over the training set. Only additional dropout layers were added to further increase the accuracy. Some experimentation was done by feeding RGB images to the model instead of grayscale, but the accuracy values were very similar. 

The test accuracy obtained is around `94%`

---
#### 3 - Testing new images

##### Acquiring New Images:

A set of eight photos were obtained through a google images search, and cropped more or less accordingly to the data set, but not exactly. The images have different resolutions and sizes so a resizing to 32x32 was performed using `cv2.resize`. The result after the resize is shown below:

![alt text][img5]

Some images are purposely difficult like the stop signal in the bottom left corner, which is an attempt to fool the A.I. of autonomous driving vehicles by putting black and white stickers on top. 


##### Performance on New Images:

The model predicted correctly 4 out of 8 images resulting in 50% accuracy rate which is a value much lower than the one from the test set. 

The results also vary (from 3 to 6 correct images) just by re-running the training, the softmax probabilities can also be completely different from run to run (for a given particular image), which is kind of unexpected. 

##### Model Certainty - Softmax Probabilities:

The priority signal and the simple stop are the only ones that are correctly identified with a high certainty.

![alt text][img6]

![alt text][img7]


The following ones are sometimes correctly predicted with low certainty:

![alt text][img8]

![alt text][img9]

![alt text][img10]

![alt text][img11]

However looking at the second/third highest softmax classes they are often of similar shape (60 sign classified as 80, roundabout and priority road both have arrows, and so on).

---
#### 4 - Improvements

- Augmenting the training set might help improve model certainty. Common data augmentation techniques include rotation, translation, zoom, flips, inserting jitter, and/or color perturbation. Tensorflow v2 seems to already have embedded tools for this purpose.

- Perform error analysis to identify which image characteristics the model has a harder time to classify.   

- Visualization of layers in the neural network, as suggested in the project, to get more insight on how the model works. Also exploring [Tensorboard](https://www.tensorflow.org/tensorboard) could be useful. 

