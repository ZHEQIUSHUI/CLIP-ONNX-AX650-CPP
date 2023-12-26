# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/axpi_pro_bsp_sdk/msp/out)
endif()

message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(CLIP-ONNX-AX650-CPP)

# 添加 FLAGS 检查代码是否有明显 bug
# include(cmake/overlook.cmake)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-aarch64-none-gnu-1.16.0)
endif()

message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIB onnxruntime)

if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
endif()

message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

option(BUILD_WITH_AX650 "build with ax650" OFF)
option(BUILD_WITH_AX620E "build with ax620e" OFF)
option(BUILD_VIDEO_DEMO "build video demo" OFF)

if(BUILD_WITH_AX620E)
    set(BUILD_WITH_AX650 ON)
endif()


if(BUILD_WITH_AX650)
    set(AXERA_TARGET_CHIP "AX650")
    add_compile_definitions(BUILD_WITH_AX650)
    add_compile_definitions(AXERA_TARGET_CHIP_AX650)
endif()

message(STATUS "BUILD_WITH_AX650 : ${BUILD_WITH_AX650}")
message(STATUS "BUILD_VIDEO_DEMO : ${BUILD_VIDEO_DEMO}")

add_library(clip STATIC
    src/Runner/BaseRunner.cpp
    src/Runner/ax_model_runner_ax650.cpp
)

add_executable(main src/main.cpp)

target_link_libraries(main clip)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main gomp ${ONNXRUNTIME_LIB})

if(BUILD_WITH_AX650)
    target_link_libraries(main ax_engine ax_interpreter ax_sys)
endif()

install(TARGETS main DESTINATION bin)

if(BUILD_WITH_AX650 AND BUILD_VIDEO_DEMO)
    add_subdirectory(ax-pipeline/third-party/RTSP)
    add_subdirectory(ax-pipeline/third-party/RtspServer)    
    add_subdirectory(ax-pipeline/third-party/mp4demux)
    add_subdirectory(ax-pipeline/examples/common/common_pipeline)
    add_executable(main_video src/main_video.cpp ax-pipeline/examples/utilities/net_utils.c ${BSP_MSP_DIR}/../sample/common/common_sys.c)

    include_directories(${BSP_MSP_DIR}/../sample/common)
    include_directories(ax-pipeline/third-party/RTSP/include)
    include_directories(ax-pipeline/third-party/mp4demux)

    target_link_directories(main_video PRIVATE ${BSP_MSP_DIR}/../../third-party/drm/lib)
    target_link_libraries(main_video PRIVATE drm)

    target_link_libraries(main_video PRIVATE clip common_pipeline)

    target_link_libraries(main_video PRIVATE ${OpenCV_LIBS})
    target_link_libraries(main_video PRIVATE gomp ${ONNXRUNTIME_LIB})

    target_link_libraries(main_video PRIVATE ax_interpreter ax_sys ax_venc ax_vdec ax_ivps ax_ive ax_engine ax_vo gomp stdc++fs)

    target_link_libraries(main_video PRIVATE RtspServer)
    target_link_libraries(main_video PRIVATE rtspclisvr Mp4Demuxer)
endif()