
include_directories(${PROJECT_SOURCE_DIR}/tests/common)
#set(tracker_SOURCES
#        tracker.cpp
#        common/VideoSource.cpp
#        common/Visualization3D.cpp
#        common/VideoSource.hpp
#        common/Visualization3D.hpp)

#add_executable(tracker ${tracker_SOURCES})

set(common_SOURCES
        common/VideoSource.cpp
        common/Visualization3D.cpp
        common/VideoSource.hpp
        common/Visualization3D.hpp)

add_library(common SHARED ${common_SOURCES})
target_link_libraries(common ${OpenCV_LIBRARIES} thymiotracker)

set(exec_SOURCES
        trackerGH.cpp
        simuArthymio.cpp)

foreach(source ${exec_SOURCES})
  # Compute the name of the binary to create
  get_filename_component(binary ${source} NAME_WE)

  # From source compile the binary and add link rules
  add_executable(${binary} ${source})
  target_link_libraries(${binary} ${OpenCV_LIBRARIES} thymiotracker common)

endforeach(source)
