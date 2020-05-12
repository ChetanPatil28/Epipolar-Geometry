# 1. Stereo-Correspondence-Matching

This project is about estimating the disparity between two images when one of the image is taken by slightly moving towards the right.
By knowing the focal length of the camera and the baseline distance, we can accurately determine the 3-d location of the point.
This project aims at implementing some of those Stereo-Matching algorithms in Python.

#### 1. BLOCK-MATCHING method for which the code is written in `BlockMatching.py`
#### 1. ENERGY MINIMIZATION VIA DYNAMIC PROGRAMMING method for which the code is written in `Dynamic.py`


Input images are Left and Right correspondingly.

LEFT Image             |  RIGHT Image
:-------------------------:|:-------------------------:
![](<Stereo_Correspondence/Inputs/tsukuba_l.png>) | ![](<Stereo_Correspondence/Inputs/tsukuba_r.png>)

 

Block-matching Disparity | Energy-Minimization via Dynamic Programming
:---------------------:|:-------------------------:
![](<Stereo_Correspondence/Outputs/tsukuba_disparity.png>)| ![](<Stereo_Correspondence/Outputs/Disparity_dynamic.png>)

###### THE BRIGHTER THE COLOR, THE NEARER THE OBJECT IS.

# 2. Epipolar-Line-Estimation

This section is about finding the Epipolar-Lines given a correspondence pair.
Essentially, we estimate the FundaMental matrix such that x1.T*F*x = 0.
The famous 8-point algorithm is used to find the F matrix.
Link to paper- https://www.cse.unr.edu/~bebis/CS485/Handouts/hartley.pdf

The resulting epipoles are shown below.

VIEW-A             |  VIEW-B
:-------------------------:|:-------------------------:
![](<Stereo_Rectification/Result/Epilines_A.png>) | ![](<Stereo_Rectification/Result/Epilines_B.png>)
