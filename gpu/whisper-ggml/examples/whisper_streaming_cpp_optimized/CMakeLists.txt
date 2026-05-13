set(TARGET whisper_streaming_cpp_optimized)
add_executable(${TARGET} whisper_streaming_cpp_optimized.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})
