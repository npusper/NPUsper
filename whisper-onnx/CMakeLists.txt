cmake_minimum_required(VERSION 3.17)
project(whisperflow-onnx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── ONNX Runtime ────────────────────────────────────────────────────────────
# Set ORT_ROOT to your onnxruntime installation path, e.g.:
#   cmake -DORT_ROOT=/path/to/onnxruntime ..
set(ORT_ROOT "" CACHE PATH "Path to ONNX Runtime installation")
set(ORT_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime headers (auto-set from ORT_ROOT)")
set(ORT_LIB_DIR     "" CACHE PATH "Path to ONNX Runtime libraries (auto-set from ORT_ROOT)")

if(ORT_ROOT)
    if(NOT ORT_INCLUDE_DIR)
        set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    endif()
    if(NOT ORT_LIB_DIR)
        set(ORT_LIB_DIR "${ORT_ROOT}/lib")
    endif()
else()
    find_path(ORT_INCLUDE_DIR onnxruntime/core/session/onnxruntime_cxx_api.h)
    find_library(ORT_LIB onnxruntime)
endif()

find_library(ORT_LIB onnxruntime HINTS ${ORT_LIB_DIR})

if(NOT ORT_LIB)
    message(FATAL_ERROR "onnxruntime library not found. Set -DORT_ROOT=<path>")
endif()

message(STATUS "ORT include: ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib:     ${ORT_LIB}")

# ── whisper ORT wrapper ─────────────────────────────────────────────────────
add_library(whisper_ort STATIC
    whisper_ort.cpp
)
target_include_directories(whisper_ort PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ORT_INCLUDE_DIR}
)
target_link_libraries(whisper_ort PUBLIC ${ORT_LIB})

# ── examples ────────────────────────────────────────────────────────────────
add_subdirectory(examples)
