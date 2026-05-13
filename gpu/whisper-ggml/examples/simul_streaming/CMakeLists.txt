set(TARGET simul_streaming)
add_executable(${TARGET} simul_streaming.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})
