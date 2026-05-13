add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PUBLIC whisper_ort)

# Only ours_streaming is needed for the X Plus NPU deploy-mode3 artifact.
add_subdirectory(ours_streaming)
