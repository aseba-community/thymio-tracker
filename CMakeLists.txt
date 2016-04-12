# Usage:
#
#    $ mkdir build && cd build
#    $ cmake ..  -DANDROID_WRAPPER=ON -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=../../../deps/OpenCV-android-sdk/sdk/native/jni/

cmake_minimum_required(VERSION 2.8)
project(ThymioTracker)

set(ANDROID_WRAPPER OFF CACHE BOOL "Compile for Android with Java wrapper")

set(ThymioTracker_SOURCES
    src/ThymioTracker.cpp
    src/GH.cpp
    src/GHscale.cpp
    src/Models.cpp
    src/Generic.cpp
    src/Grouping.cpp
    src/BlobInertia.cpp
    src/Landmark.cpp
    src/Robot.cpp
    src/ThymioTracker.h
    src/Generic.hpp
    src/GH.hpp
    src/GHscale.hpp
    src/Models.hpp
    src/Grouping.hpp
    src/BlobInertia.hpp
    src/Landmark.hpp
    src/Robot.hpp
    )

if(ANDROID_WRAPPER)
    #SET(JAVA_INCLUDE_PATH "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/include")
    #SET(JAVA_AWT_LIBRARY "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/libawt.dylib")
    #SET(JAVA_JVM_LIBRARY "/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib")
    
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
        android/java/ch/epfl/cvlab/thymiotracker/ThymioTracker.java)
    set(CMAKE_JAVA_INCLUDE_PATH
        ${CMAKE_JAVA_INCLUDE_PATH}
        "${OpenCV_ANDROID_SDK}/java/src")
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

    include_directories(
        ${JAVA_INCLUDE_PATH}
        ${_stubDir})

    INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )

    set(ThymioTracker_SOURCES
        ${ThymioTracker_SOURCES}
        ch_epfl_cvlab_thymiotracker_ThymioTracker.h
        android/jni/ch_epfl_cvlab_thymiotracker_ThymioTracker.cpp)

else(ANDROID_WRAPPER)
    find_package(OpenCV REQUIRED)

    INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" )
    
endif(ANDROID_WRAPPER)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-long-long -Wno-vla -pedantic")

# add_subdirectory(brisk)

include_directories(${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/include
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
