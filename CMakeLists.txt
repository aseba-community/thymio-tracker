# Usage:
#
#    $ mkdir build && cd build
#    $ cmake ..  -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=../../../deps/OpenCV-android-sdk/sdk/native/jni/

cmake_minimum_required(VERSION 2.8)
project(ARThymio)

set(ANDROID_WRAPPER ON CACHE BOOL "Compile Android/Java wrapper")

set(ARThymio_SOURCES
    arthymio.cpp)

if(ANDROID_WRAPPER)
    
    # OpenCV
    find_host_package(OpenCV REQUIRED)
    set(OpenCV_ANDROID_SDK "${OpenCV_CONFIG_PATH}/../../" CACHE FILEPATH "Path to OpenCV4Android SDK (usually ending in <path>/sdk/)")
    
    # Java
    find_host_package(Java REQUIRED)
    find_host_package(JNI REQUIRED)
    include(UseJava)

    # set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")

    # JAR
    set(ARThymio_JAVA_SOURCES
        android/java/ch/epfl/cvlab/arthymio/ARThymio.java)
    set(CMAKE_JAVA_INCLUDE_PATH
        ${CMAKE_JAVA_INCLUDE_PATH}
        "${OpenCV_ANDROID_SDK}/java/src")
    add_jar(arthymio_android SOURCES ${ARThymio_JAVA_SOURCES})
    get_target_property(_classDir arthymio_android CLASSDIR)
    get_target_property(_jarFile arthymio_android JAR_FILE)

    # JNI header
    set(_stubDir ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_command(
        OUTPUT ch_epfl_cvlab_arthymio_ARThymio.h
        DEPENDS ${_jarFile}
        COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
            -classpath ${_classDir}:${CMAKE_JAVA_INCLUDE_PATH}
            -d ${_stubDir}
            ch.epfl.cvlab.arthymio.ARThymio)

    include_directories(
        ${JAVA_INCLUDE_PATH}
        ${_stubDir})

    set(ARThymio_SOURCES
        ${ARThymio_SOURCES}
        ch_epfl_cvlab_arthymio_ARThymio.h
        android/jni/ch_epfl_cvlab_arthymio_ARThymio.cpp)
    
endif(ANDROID_WRAPPER)

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
add_library(arthymio MODULE ${ARThymio_SOURCES})
target_link_libraries(arthymio ${OpenCV_LIBRARIES})
