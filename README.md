**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/hog_car_features.jpg
[image4]: ./output_images/hog_notcar_features.jpg
[image5]: ./output_images/test_image.jpg
[image6]: ./output_images/test_image_boxed.jpg
[image7]: ./output_images/heat1.png
[image8]: ./output_images/heat2.png
[image9]: ./output_images/heat3.png
[image10]: ./output_images/heat4.png
[image11]: ./output_images/heat5.png
[image12]: ./output_images/heat6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features and color histograms from the training images.

The code for this step is contained in the second and 3th code cell of the IPython notebook (Vehicle_Detection.ipynb)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car image][image1]
Vehicle

![not car image][image2]
Non Vehicle

I experiented with different settngs for the HOG features and coor histograms. Testing them when training a SVM I settled on the following parameters:

Color Space = 'YCrCb'

HOG orientations = 15

HOG pixels per cell = 16

HOG cells per block = 2

HOG Channel = "ALL"

Spatial binning dimensions = (16, 16)

Number of histogram bins = 16

Here is an example using the above settings:

![car image][image1]
![hog features][image3]

#### 2. How I settled on my final choice of HOG parameters.

I tried various combinations of parmeters. When running the classifier agains them, this where te ones that performed best for me.
I choose 16 HOG pixels per cell to improve speed as it did not seem to hurt performance.
The transform_sqrt setting is set to True, which gave better results.
I also set the block_norm to L1 The scikit learn documentation advices to use the newer L2-Hys block_norm, but the L1 gave me better accuracy when training.

#### 3. Training a classifier using your HOG features and color features.

The code for this step is contained in the 6th code cell of the IPython notebook (Vehicle_Detection.ipynb)

I trained a linear SVM and tried different settings to improve performance.
The training is done using HOG features, color histograms and spatial binning.
Tuning the C paramater made a noticable difference in my accuracy. It increased my accuracy from +/-98% to +99%.

### Sliding Window Search

#### 1. Implementation sliding window search.

The code for this step is contained in the 7th and 8th code cell of the IPython notebook (Vehicle_Detection.ipynb)

I tried to optimize for performance and accuracy.
The `find_cars` function only calculates the HOG feature once for each window. This is much faster than calculating it for eacht block.
In my first try, the classifier was not as precise and I had to do about 10 calls to the `find_cars` function. This caused the pipeline to be unacceptable slow.
After improving the classifier I cut out some frames from the video that gave me the most problems. I used those frames together with the already available test picures to optimize the pipeline.
After trying several different amount of windows with different setting I ended up with the following:

Amount of windows needed: 3

##### Window Settings:

Window 1: ystart 360, ystop 656, scale 1.5

Window 2: ystart 360, ystop 656, scale 1.6

Window 3: ystart 360, ystop 656, scale 1.8

#### 2. Examples of test images to demonstrate the pipeline.

Here is an example using the above settings:

![original image][image5]
![original image with boxes][image6]

---

### Video Implementation

#### 1. Link to the final video output.
Here's a [link to my video result](https://youtu.be/_nmful_9fqY)


#### 2. Filtering false positives and combining overlapping bounding boxes.

The code for this step is contained in the 8th and 9th code cell of the IPython notebook (Vehicle_Detection.ipynb)

I kept a list of all positive detection in a frame. With the `add_heat` function I added boxed to a heatmap.
With the `apply_threshold` function I filter out boxes with a minimum amount of overlap.

Using the `scipy.ndimage.measurements.label()` I indentified blob in the heatmap, which I classified as vehicles.

The `draw_labeled_bboxes` function draws a box around each area detected as a car.

Here's an example result showing the heatmap from a series of frames of video:

### Here are six frames their corresponding heatmaps and the bounding boxes drawn:

![heat map][image7]

![heat map][image8]

![heat map][image9]

![heat map][image10]

![heat map][image11]

![heat map][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found it very difficult to get all the parameters right, so it would detect cars but not excluded false positives.
The SVM C parameter made a big difference for me in excluding false positives.

The pipeline seems to fail when there are big brighness differences in small patches of an image.

If I am going to pursue this project further I would try a deep learning aproach. I think a neural network with convolutional layers might be better in learning to recognize cars.

