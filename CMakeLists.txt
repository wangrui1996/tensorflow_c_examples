cmake_minimum_required(VERSION 3.6)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/lib)

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)


if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
    make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
endif()

if(UNIX AND NOT APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)
elseif(APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-darwin-x86_64-1.14.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(tensorflow/lib)
elseif(WIN32)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-windows-x86_64-1.14.0.zip ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(tensorflow/lib)
    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll COPYONLY)
    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/test/tensorflow.dll COPYONLY)
endif()

include_directories(include)

find_package(OpenCV REQUIRED)
add_library(tensorflow_api STATIC src/batch_interface_object.cpp)
target_link_libraries(tensorflow_api tensorflow ${OpenCV_LIBS})

configure_file(models/frozen_inference_graph.pb ${CMAKE_CURRENT_BINARY_DIR}/test/frozen_inference_graph.pb COPYONLY)

enable_testing()
add_subdirectory(test)
