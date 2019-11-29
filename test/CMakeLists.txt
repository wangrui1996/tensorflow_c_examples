# ---[ OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_executable(test_detection test_detection.cpp)
target_link_libraries(test_detection tensorflow_api ${OpenCV_LIBS})
