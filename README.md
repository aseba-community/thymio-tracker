# Thymio Tracker

Thymio tracker is a C++ visual tracker for AR Thymio.
The only dependency is OpenCV. The code has been tested using OpenCV-3.1.0.

\SECTION Default Tracker Functionalities

Thymio tracker can be divided into two main tracking applications:
 - robot tracking, 
 - landmarks tracking.

As the main objective is to do AR, the tracker has to deliver the 3D pose of 
the camera with respect to the robot and landmarks. To do so the camera has to
be calibrated.

/subsection Camera Calibration

ThymioTracker uses the standard openCV xml calibration files. To generate such
a calibration, use either openCV's example codes, or use the ./calibrate binary.
The ./calibrate binary uses the online capture, it detects the assymetric grid
pattern pattern-a4.pdf which can be found in the data folder, if a few detection
are succefull in a row, then the image is stored to compute the calibration, and
this until enough calibration images are acquired.

Once enough images are acquired the calibration is computed and stored in 
the xml output file which path has to be provided as only parameter to the program.
The path to the external calibration can then be defined in the Config.xml file
which is loaded by the ThymioTracker object (See the Config.xml example file).

/subsection Robot tracker




/subsection Landmarks tracker






/SECTION Definition of new tracking models

This section explains how to use the tracker for new landmarks or for a different
robot model.

/subsection New robot model

/subsubsection New blob configuration

As the geometric hashing method used for the robot detection uses the blob configuration,
the hash table of the geometric hashing has to be updated. To do so use the ./trainGH program.

/subsubsection New appearance model

As the tracking works using a set of planar surfaces which texture is learned, if the appearance
of the robot changes or if one wants to use more planar surfaces, then the texture of the planar
surfaces will have to be updated.

To update the surfaces texture model, use the ./onBoard program. This program will need as input 
an image sequence and the corresponding poses of the robot (the robot to camera homogeneous 
transformation matrix). The model which is currently defined in the modelSurfaces.xml.gz file
has been acquired using aruco library and an AR board (aruco has also as only dependency OpenCV).

/subsection New landmarks

To be able to track new landmark, use the ./landmark program from the tools folder.
Provide as first parameter the input image of the new landmark and as second parameter
the xml.gz file to be outputed and then added to the Config.xml file loaded by the 
ThymioTracker object.






