cmake_minimum_required(VERSION 3.0)

project(CLIPQT_VideoAnalyst LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

enable_language(C CXX)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

# if(ANDROID)
# set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
# if (ANDROID_ABI STREQUAL "armeabi-v7a")
# set(ANDROID_EXTRA_LIBS
# ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
# ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
# endif()
# endif()

# bsp
if(NOT BSP_MSP_DIR)
  set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/../../axpi_pro_bsp_sdk/msp/out)
endif()
message(STATUS "BSP_MSP_DIR Path: ${BSP_MSP_DIR}")

find_package(Qt6 COMPONENTS Widgets)

if(Qt6_FOUND)
  set(QTWidget Qt6::Widgets)
else()
  find_package(Qt5 COMPONENTS Widgets REQUIRED)
  set(QTWidget Qt5::Widgets)
endif()

if(ANDROID)
  add_library(CLIPQT_VideoAnalyst SHARED
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
  add_executable(CLIPQT_VideoAnalyst
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    myqlabel.h
    style/DarkStyle.cpp
    style/DarkStyle.h

    clip/cqdm.cpp
    clip/Runner/BaseRunner.cpp
    clip/Runner/ax_model_runner_ax650.cpp
    ax-pipeline/examples/utilities/net_utils.c
    ${BSP_MSP_DIR}/../sample/common/common_sys.c
  )
endif()

target_link_libraries(CLIPQT_VideoAnalyst PRIVATE ${QTWidget})

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
if(NOT ONNXRUNTIME_DIR)
  set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime)
endif()

message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
target_link_directories(CLIPQT_VideoAnalyst PRIVATE ${ONNXRUNTIME_DIR}/lib)
set(ONNXRUNTIME_LIB onnxruntime)

if(NOT OpenCV_DIR)
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../third_party/opencv-mobile-4.6.0-ubuntu-2004/lib/cmake/opencv4)
endif()

message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BSP_MSP_DIR}/include)
target_link_directories(CLIPQT_VideoAnalyst PRIVATE ${BSP_MSP_DIR}/lib)


set(AXERA_TARGET_CHIP "AX650")
add_compile_definitions(BUILD_WITH_AX650)
add_compile_definitions(AXERA_TARGET_CHIP_AX650)

target_link_libraries(CLIPQT_VideoAnalyst PRIVATE ${OpenCV_LIBS})
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE gomp ${ONNXRUNTIME_LIB})

add_subdirectory(ax-pipeline/third-party/RTSP)
add_subdirectory(ax-pipeline/third-party/RtspServer)
add_subdirectory(ax-pipeline/third-party/mp4demux)
add_subdirectory(ax-pipeline/examples/common/common_pipeline)

include_directories(${BSP_MSP_DIR}/../sample/common)
include_directories(ax-pipeline/third-party/RTSP/include)
include_directories(ax-pipeline/third-party/mp4demux)

target_link_directories(CLIPQT_VideoAnalyst PRIVATE ${BSP_MSP_DIR}/../../third-party/drm/lib)
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE drm)
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE common_pipeline)
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE ${OpenCV_LIBS})
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE gomp ${ONNXRUNTIME_LIB})
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE ax_interpreter ax_sys ax_venc ax_vdec ax_ivps ax_ive ax_engine ax_vo gomp stdc++fs)
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE RtspServer)
target_link_libraries(CLIPQT_VideoAnalyst PRIVATE rtspclisvr Mp4Demuxer)
