# README #

This is project to identify lanes on the road and display lane markers. The basic idea behind this project is to use these lane markers possibly in some cool project extensions such as self-driving car or vehicular guidance systems.

### Help for the project ###
The project is a part of the tutorial found on a [KDNuggets](http://www.kdnuggets.com/2017/07/road-lane-line-detection-using-computer-vision-models.html). I found it interesting and as a beginner step to implement a much grander vehicule guidance system. This is first part of the project. I intend to doing new projects and maybe club them all together to make a full system.
A single page details of the project can be found as a gist on [Github](https://github.com/vijay120/KDNuggets/blob/master/2016-12-04-detecting-car-lane-lines-using-computer-vision.md), where it was originally posted, all thanks to Vijay Ramakrishnan.

### Steps involved in the project: ###
* We begin processing images. Video is just a collection of frames of images. So once we can figure out to analyze and process the image, we can just pass each frame of the video and process them.
* The first step involved in the processing of images is converting color images to greyscale. It's easier to work with single color channel than multi-color channel.
* Next we apply Gaussian Smoothing function to this greyscale image. This is needed for the Edge Detection.
* Next, we need to detect edges for lane detection since the contrast between the lane and the surrounding road surface provides us with useful information on detecting the lane lines. 
    We do this using [Canny Edge Detection](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html). Canny edge detection is an operator that uses the horizontal and vertical gradients of the pixel values of an image to detect edges.
We will then filter out the rest of the edges by selecting a polygon that will only be selecting the edges corresponding to the lanes/road.
* Next, we will apply Hough Transform. I checked out this video for [Hough Transform](https://www.youtube.com/watch?v=4zHbI-fFIlI). 
    The Hough transformation converts a “x vs. y” line to a point in “gradient vs. intercept” space. Points in the image will correspond to lines in hough space. An intersection of lines in hough space will thus correspond to a line in Cartesian space. Using this technique, we can find lines from the pixel outputs of the canny edge detection output.
* Next we will draw lines. The hough transform gives us small lines based on the intersections in hough space. Now, we can take this information and construct a global left lane line and a right lane line.
* After this we will extrapolate this image(lines) to the original image.
* Add this to the video.