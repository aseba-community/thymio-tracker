
include_directories(${PROJECT_SOURCE_DIR}/tests/common)
set(tracker_SOURCES
        tracker.cpp
        common/VideoSource.cpp
        common/Visualization3D.cpp
        common/VideoSource.hpp
        common/Visualization3D.hpp)

add_executable(tracker ${tracker_SOURCES})
target_link_libraries(tracker ${OpenCV_LIBRARIES} arthymio)
