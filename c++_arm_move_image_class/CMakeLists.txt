project(arm_move_image_class)
cmake_minimum_required(VERSION 2.8)

aux_source_directory(. SRC_LIST)




if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0015 OLD)  
endif(COMMAND cmake_policy)


find_package(OpenCV 3.1 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nodelet REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(actionlib REQUIRED)
find_package(tf REQUIRED)

find_package(Boost 1.4 COMPONENTS program_options REQUIRED)

include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})
include_directories(${cv_bridge_INCLUDE_DIRS})
include_directories(${image_transport_INCLUDE_DIRS})
include_directories(${geometry_msgs_INCLUDE_DIRS})
include_directories(${nodelet_INCLUDE_DIRS})
include_directories(${nav_msgs_INCLUDE_DIRS})
include_directories(${message_filters_INCLUDE_DIRS})
include_directories(${stereo_msgs_INCLUDE_DIRS})
include_directories(${move_base_INCLUDE_DIRS})
include_directories(${move_base_msgs_INCLUDE_DIRS})
include_directories(${actionlib_INCLUDE_DIRS})
include_directories(${tf_INCLUDE_DIRS})

include_directories(/usr/include/eigen3)

include_directories(${Boost_INCLUDE_DIR})

add_definitions(-std=c++0x) # -m64)


add_executable(move_rebecca  src/main.cpp)
target_link_libraries(move_rebecca ${OpenCV_LIBS}
        ${roscpp_LIBRARIES} ${std_msgs_LIBRARIES} ${sensor_msgs_LIBRARIES}
        ${cv_bridge_LIBRARIES} ${image_transport_LIBRARIES} ${message_filters_LIBRARIES} ${stereo_msgs_LIBRARIES})

