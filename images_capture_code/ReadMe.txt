this contains the instructions for running the python code for generating a dataset from the intel real sense camera.

When this code is run, the realsense camera should also be paralleling running by opening a new terminal and typing:

//roslaunch realsense2_ros_camera rs_aligned_depth.launch >> error

"

Start the camera node
To start the camera node in ROS, install rgbd_launch:

sudo apt-get install ros-kinetic-rgbd-launch
Then type:

roslaunch realsense2_camera rs_camera.launch
This will stream all camera sensors and publish on the appropriate ROS topics.

Other stream resolutions and frame rates can optionally be provided as parameters to the 'rs_camera.launch' file.

"


Running instructions for dataset generation:

python save_image.py
