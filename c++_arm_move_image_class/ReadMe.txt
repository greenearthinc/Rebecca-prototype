1. This contains the c++ code to send a robot arm motion command through serial communication to arduino. For example in line 38, makes sending the string "q30" on the serial port to be picked up by ardiuno.
2. Make sure that in line 30, the serial port is /dev/ttyACM0 OR this can sometimes be different too, find this out by opening the arduino GUI software and goto tools->port-> PORT PATH

3. After making the required changes in this code, Run the following:

cd /home/greenearth/Desktop/prototype_code/c++_arm_move_image_class/build
cmake ..
make
./move_rebecca
