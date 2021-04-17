# Learning Human Touch Interaction with Convolutional Neural Networks
A repository to save the works done during my research internship at [Gentiane Venture's Laboratory](http://web.tuat.ac.jp/~gvlab/), Tokyo Univ. of Agriculture and Technology, Japan.

The research is about using Convolutional Neural Networks (CNN) to classify several types of touch interaction (poke, scratch, gentle stroke, hard press, and neutral) from humans by learning the data pattern from a force sensor. This sensor converts human touch into 3-dimensional force data.

## The Data <br>
> * Acquired the data by using pySerial for every 0.02 second (50 fps). Ten people did every touch interaction 30 times to the sensor and recorded the data. <br>
> * Inferred a suitable threshold to differentiate positive signals from noise and an appropriate number of frames for one positive event (sample) from the data statistics. <br>
> * Preprocessed the data by interpolating every data sample into 40 frames and normalizing them to reduce the effect of zero-bias noise. <br>
> * The dataset is not being shared here since it is belong to the individuals in the lab. A process to cover the identity of the subjects to protect their privacy will be done soon and the dataset will then be released.

## The Model
> * Trained the CNN model with the data using the Keras framework in Python. Adam optimizer was used for the training with lr=1e-4. <br>
> * This is the model's architecture <br>
![arch plot](https://github.com/eraraya-ricardo/gvlab-research-internship/blob/main/architecture_plot.png)

# Real-time Testing
> * For real-time inference, applied a moving window algorithm to record the last 40 frames (1 window = 40 frames) of the normalized sensor's data every time the sensor sends a new force data. <br>
> * The CNN model then predicted the type of touch received by the sensor for every new window. <br>
> * The model reached **88% accuracy during real-time deployment**. <br>
