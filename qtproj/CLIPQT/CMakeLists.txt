cmake_minimum_required(VERSION 3.5)

project(CLIPQT LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(Qt6 COMPONENTS Widgets)
if(Qt6_FOUND)
  set(QTWidget Qt6::Widgets)
else()
  find_package(Qt5 COMPONENTS Widgets REQUIRED)
  set(QTWidget Qt5::Widgets)
endif()

if(ANDROID)
  add_library(CLIPQT SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    style/DarkStyle.cpp
    style/DarkStyle.h

    clip/cqdm.cpp
    clip/Runner/BaseRunner.cpp
    clip/Runner/ax_model_runner_ax650.cpp
  )
else()
  add_executable(CLIPQT
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui

    internal_func.cpp
    internal_func.hpp

    style/DarkStyle.cpp
    style/DarkStyle.h

    clip/cqdm.cpp
    clip/Runner/BaseRunner.cpp
    clip/Runner/ax_model_runner_ax650.cpp
  )
endif()

target_link_libraries(CLIPQT PRIVATE ${QTWidget})

# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/../../axpi_pro_bsp_sdk/msp/out)
endif()

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime)
endif()

message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
target_link_directories(CLIPQT PRIVATE ${ONNXRUNTIME_DIR}/lib)
set(ONNXRUNTIME_LIB onnxruntime)

if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../third_party/opencv-mobile-4.6.0-ubuntu-2004/lib/cmake/opencv4)
endif()

message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BSP_MSP_DIR}/include)
target_link_directories(CLIPQT PRIVATE ${BSP_MSP_DIR}/lib)

option(BUILD_WITH_AX650 "build with ax650" OFF)

if(BUILD_WITH_AX650)
    add_compile_definitions(BUILD_WITH_AX650)
endif()

message(STATUS "BUILD_WITH_AX650 : ${BUILD_WITH_AX650}")
if(BUILD_WITH_AX650)
    target_link_libraries(CLIPQT PRIVATE ax_engine ax_interpreter ax_sys)
endif()

target_link_libraries(CLIPQT PRIVATE ${OpenCV_LIBS})
target_link_libraries(CLIPQT PRIVATE gomp ${ONNXRUNTIME_LIB})
