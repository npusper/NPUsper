set(TARGET simul_whisper)
add_executable(${TARGET} simul_whisper.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})
