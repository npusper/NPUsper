add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PUBLIC whisper_ort)

# Only ours_streaming is needed for NPU experiments
add_subdirectory(ours_streaming)

# Other systems (kept for reference, not built)
# add_subdirectory(whisper_streaming)
add_subdirectory(simul_whisper)
# add_subdirectory(simul_streaming)
add_subdirectory(verify_cpp)
