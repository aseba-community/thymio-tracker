include_directories(${PROJECT_SOURCE_DIR}/tests/common)

set(tools_SOURCES
        learnSurfaces.cpp
        calibrate.cpp
        trainGH.cpp
        landmark.cpp)

foreach(source ${tools_SOURCES})
  # Compute the name of the binary to create
  get_filename_component(binary ${source} NAME_WE)

  # From source compile the binary and add link rules
  add_executable(${binary} ${source})
  target_link_libraries(${binary} ${OpenCV_LIBRARIES} thymiotracker common)

endforeach(source)

