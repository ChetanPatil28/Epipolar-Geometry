# Stereo-Correspondence-Matching

This project is about estimating the disparity between two images when one of the image is taken by slightly moving towards the right.
By knowing the focal length of the camera and the baseline distance, we can accurately determine the 3-d location of the point.

#### 1. One method of estimating disparity is called the BLOCK-MATCHING method for which the code is written.
Input images are Left and Right correspondingly.

LEFT Image             |  RIGHT Image
:-------------------------:|:-------------------------:
![](<Inputs/tsukuba_l.png>) | ![](<Inputs/tsukuba_r.png>)

 

Inferred Disparity
:---------------------:
![](<Outputs/tsukuba_disparity.png>)

THE BRIGHTER THE COLOR, THE NEARER THE OBJECT IS.
