# Behavioural Cloning Project


## Useful information

**Goal of the project:** Use a simulator to gather human driving 
behaviour data around a track. Use the gathered data to train a 
deep neural network to drive a car autonomously around a test track
in a simulator. 


**Files of interest:** 

- model.py (code for the proposed neural network and data processing)
- drive.py (Udacity's proposed autonomous driving controller/script)
- model.h5 (The saved Keras model after training)
- video.mp4 (A generated video file for performance evaluation)

---

## Network architecture and training

### Convolutional Network Architecture

The network architecture proposed as a solution to this project is 
NVidia's deep CNN architecture as proposed in their 
[End to end learning for Self-driving cars paper](https://arxiv.org/pdf/1604.07316.pdf),
with changes applied to the input layer to reflect the image size used for training in 
this project. Other layers were also added, as seen in the table below.

![NVidia's model architecture](./examples/placeholder.png) 

The original model architecture. The resulting modified architecture based on the 
NVidia model is presented in the table below.

| Layer         		| Description	        					    | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB images                          |
| Normalization         | Lambda normalization layer                    |
| Cropping, 2D          | Crop the image to filter out unnecessary info |
| Convolution 5x5     	| 2x2 stride, valid padding                 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding                 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding                 	|
| RELU					|												|
| Dropouts              |                                               |
| Convolution 3x3     	| 1x1 stride, valid padding	                    |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding	                    |
| RELU					|												|
| Dropouts              |                                               |
| Flatten       	    | Flattening the feature maps   				|
| Fully connected		| outputs 100, L2 weights regularization   		|
| RELU					|												|
| Dropouts              |                                               |
| Fully connected		| outputs 50, L2 weights regularization  		|
| RELU					|												|
| Dropouts              |                                               |
| Fully connected		| outputs 10, L2 weights regularization   		|
| RELU					|												|
| Logits        		| outputs 1 steering angle prediction   		|


As seen in the table, dropout layers with probability of 0,5 were added to 
reduce the risk of overfitting the model and provide a better CNN result.
Some non-linearity is introduced to the model by using RELUs as activation 
functions for each layer, both for the feature maps and in the dense, fully 
connected part of the model. Also, L2 regularization was introduced in the 
fully connected layers to help the model prevent overfitting as well.

An Adam optimizer was used for training.


### Data preprocessing and agumentation

data rationale (how and why the data was gathered) analog control via mouse etc
data count, some examples
data normalization
agumentation
other
train/test splits

### Training Approach

generator
data shuffling
