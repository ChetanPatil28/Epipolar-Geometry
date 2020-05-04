# Stereo-Correspondence-Matching

This project is about estimating the disparity between two images when one of the image is taken by slightly moving towards the right.
By knowing the focal length of the camera and the baseline distance, we can accurately determine the 3-d location of the point.
This project aims at implementing some of those Stereo-Matching algorithms in Python.

#### 1. BLOCK-MATCHING method for which the code is written in `BlockMatching.py`
#### 1. ENERGY MINIMIZATION VIA DYNAMIC PROGRAMMING method for which the code is written in `Dynamic.py`


Input images are Left and Right correspondingly.

LEFT Image             |  RIGHT Image
:-------------------------:|:-------------------------:
![](<Inputs/tsukuba_l.png>) | ![](<Inputs/tsukuba_r.png>)

 

Block-matching Disparity | Energy-Minimization via Dynamic Porgramming
:---------------------:|:-------------------------:
![](<Outputs/tsukuba_disparity.png>)| ![](<Outputs/Disparity_dynamic.png>)

###### THE BRIGHTER THE COLOR, THE NEARER THE OBJECT IS.
