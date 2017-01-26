# Usage:
#
#    $ mkdir build && cd build
#    $ cmake ..  -DANDROID_WRAPPER=ON -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=../../../deps/OpenCV-android-sdk/sdk/native/jni/
#    or
#    $cmake ../source  -DANDROID_WRAPPER=ON -DCMAKE_TOOLCHAIN_FILE=../source/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=~/Libs/OpenCV-android-sdk/sdk/native/jni/
#    or
#    $cmake same_previous -DJAVA_INCLUDE_PATH=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/include 
#           -DJAVA_AWT_LIBRARY=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/libawt.dylib
#           -DJAVA_JVM_LIBRARY=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib

cmake_minimum_required(VERSION 2.8)
project(ThymioTracker)

set(ANDROID_WRAPPER OFF CACHE BOOL "Compile for Android with Java wrapper")

set(ThymioTracker_SOURCES
    src/ThymioTracker.h
    src/ThymioTracker.cpp
    src/GH.hpp
    src/GH.cpp
    src/GHscale.hpp
    src/GHscale.cpp
    src/Models.hpp
    src/Models.cpp
    src/Generic.hpp
    src/Generic.cpp
    src/Grouping.hpp
    src/Grouping.cpp
    src/BlobInertia.hpp
    src/BlobInertia.cpp
    src/Landmark.hpp
    src/Landmark.cpp
    src/Robot.hpp
    src/Robot.cpp
    src/TrackingFcts.hpp
    src/TrackingFcts.cpp
    src/Calibrator.hpp
    src/Calibrator.cpp
    dashel/dashel/dashel-common.cpp
    dashel/dashel/dashel-posix.cpp
    aseba/common/utils/FormatableString.cpp
    aseba/common/utils/utils.cpp
    aseba/common/utils/HexFile.cpp
    aseba/common/utils/BootloaderInterface.cpp
    aseba/common/msg/msg.cpp
    aseba/common/msg/NodesManager.cpp
    )

if(ANDROID_WRAPPER)
    SET(JAVA_INCLUDE_PATH "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/include")
    SET(JAVA_AWT_LIBRARY "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/libawt.dylib")
    SET(JAVA_JVM_LIBRARY "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib")
    
    # OpenCV
    find_host_package(OpenCV REQUIRED)
    set(OpenCV_ANDROID_SDK "${OpenCV_CONFIG_PATH}/../../" CACHE FILEPATH "Path to OpenCV4Android SDK (usually ending in <path>/sdk/)")
    
    # Java
    find_host_package(Java REQUIRED)
    find_host_package(JNI REQUIRED)
    include(UseJava)

    # set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")

    # JAR
    set(ThymioTracker_JAVA_SOURCES
        android/java/ch/epfl/cvlab/thymiotracker/ThymioTracker.java
        android/java/ch/epfl/cvlab/calibrator/Calibrator.java)
    set(CMAKE_JAVA_INCLUDE_PATH
        ${CMAKE_JAVA_INCLUDE_PATH}
        "${OpenCV_ANDROID_SDK}/java/src"
        )
    add_jar(thymiotracker_android SOURCES ${ThymioTracker_JAVA_SOURCES})
    get_target_property(_classDir thymiotracker_android CLASSDIR)
    get_target_property(_jarFile thymiotracker_android JAR_FILE)

    # JNI header
    set(_stubDir ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_command(
        OUTPUT ch_epfl_cvlab_thymiotracker_ThymioTracker.h
        DEPENDS ${_jarFile}
        COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
            -classpath ${_classDir}:${CMAKE_JAVA_INCLUDE_PATH}
            -d ${_stubDir}
            ch.epfl.cvlab.thymiotracker.ThymioTracker)

    add_custom_command(
        OUTPUT ch_epfl_cvlab_calibrator_Calibrator.h
        DEPENDS ${_jarFile}
        COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
            -classpath ${_classDir}:${CMAKE_JAVA_INCLUDE_PATH}
            -d ${_stubDir}
            ch.epfl.cvlab.calibrator.Calibrator)

    include_directories(
        ${JAVA_INCLUDE_PATH}
        ${_stubDir})

    #INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )

    set(ThymioTracker_SOURCES
        ${ThymioTracker_SOURCES}
        ch_epfl_cvlab_thymiotracker_ThymioTracker.h
        ch_epfl_cvlab_calibrator_Calibrator.h
        android/jni/ch_epfl_cvlab_thymiotracker_ThymioTracker.cpp
        android/jni/ch_epfl_cvlab_calibrator_Calibrator.cpp)

else(ANDROID_WRAPPER)
    find_package(OpenCV REQUIRED)

    #INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
    
endif(ANDROID_WRAPPER)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-long-long -Wno-vla -pedantic")

# add_subdirectory(brisk)

include_directories(${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/dashel
                    # ${PROJECT_SOURCE_DIR}/brisk/include
                    # ${PROJECT_SOURCE_DIR}/brisk/agast/include
                    ${OpenCV_INCLUDE_DIRS})

if(ANDROID_WRAPPER)
    
    add_library(thymiotracker MODULE ${ThymioTracker_SOURCES})
    
else(ANDROID_WRAPPER)
    
    add_library(thymiotracker SHARED ${ThymioTracker_SOURCES})
    add_subdirectory(tests)
    add_subdirectory(tools)

endif(ANDROID_WRAPPER)

target_link_libraries(thymiotracker ${OpenCV_LIBRARIES})
