add_executable(verify_cpp verify_cpp.cpp)
target_link_libraries(verify_cpp PRIVATE common whisper_ort)
