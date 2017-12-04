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


### Data preprocessing and agumentation

The data collecting and (pre)processing is probably the most significant part of this project.
For a nice, balanced dataset, I opted for a few laps of center-lane driving around the test track
in the Udacity car simulator. As the steering angle is constantly recorded during data collection,
it seemed appropriate to have as many different values in the steering angle's range as possible.
In order to do that, I used the mouse controls to steer the car more precisely and get the widest range of
different steering angles. This means that the model can learn to steer precisely,
instead of steering fully to the left (or right) all the time to correct its heading.

To balance the dataset:
- an equal number of laps driven in both directions of the track were recorded,
- all the images in the dataset were flipped around the vertical axis to
increase the data points count.
- a recovery lap was recorded to help the model recover should it end
off the road in any case.

Normalization on the dataset was done in the input lambda layer,
by dividing each pixel's color value by 255,
then subtracting 0.5 to center the dataset around zero.

Random generated RGB noise was added to the image to prevent the model
from overfitting on the test track images.
- since we're training the model on RGB images, it would be interesting
to see if colour shifts can improve the model in a further point of time too.

![Noise example](examples/noise_image.png)

An example of the RGB noise added to a photo.

Each frame of the dataset is cropped inside the model to reduce
the model's size and remove unnecessary image features from influencing
the model, thus forcing the model to learn to predict steering angles
only by looking at the road ahead.

Images from the left and right cameras were also used for training,
with a corresponding steering angle corrective bias of 0.25 added to
each left camera image measurement, and -0.25 to each right camera image
 measurement.

#### Important numbers
- Total data points (steering measurements, paired with corresponding images): **X**
- A **70/30 train/test split** was used.
- Training dataset: **Y** data points
- Validation dataset: **Z** data points

### Training Approach

Training was done in 3 epochs, with a batch size of 3072 data points.
The training dataset was passed on to the model via a generator function
to save system resources, although the batch size was tuned to maximize
RAM usage as much as possible. The data is shuffled each time, prior to
sending each new batch for training, so data order should have no
impact on training whatsoever.

An Adam optimizer was used for training, and the loss was calculated
as a mean-squared error.

The model finished training with a **training loss of 0.1386**, and a
**validation loss of 0.1148**. I chose to train the final model for **3 epochs**
only since I observed that the loss was almost stagnating after around that
number of epochs, so no serious improvements are observed afterwards.

The final model performs really well on the test track, where it is
clear that the analog steering measurements aid the model in driving around
and following the center of the track. Small steering angles tune the car's
position even on straights, while maintaining a speed of **15MPH**.

A video showing the model's performance on the test track can be found
[here](video.mp4).

### Further optimization (TODOs)
- Improve the model's ability to handle twisted, tight turns by fine-tuning
the model on a new dataset, recording only sharp turns.
- Try shifting the colour on some images to see if the model would
generalize.
- Export the final model in a JSON form and save the weights,
since KERAS is notorious for incompatibility between recent versions's
.h5 files. I already encountered a problem of this sort, working on 2 PCs
in parallel.
- Push the speed of the vehicle to the maximum (30MPH).