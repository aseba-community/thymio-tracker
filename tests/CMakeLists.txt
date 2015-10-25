
set(tracker_SOURCES
        tracker.cpp
        VideoSource.cpp
        Visualization3D.cpp)

add_executable(tracker ${tracker_SOURCES})
target_link_libraries(tracker ${OpenCV_LIBRARIES} arthymio)
