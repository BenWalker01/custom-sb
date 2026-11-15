cmake_minimum_required(VERSION 3.16)
project(sweatbox_stt LANGUAGES C CXX)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

if(NOT DEFINED WHISPER_CPP_DIR)
    message(FATAL_ERROR "Set WHISPER_CPP_DIR to the whisper.cpp checkout")
endif()

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${WHISPER_CPP_DIR}" whisper-build EXCLUDE_FROM_ALL)

add_executable(stt-worker worker.cpp)
set_target_properties(stt-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_features(stt-worker PRIVATE cxx_std_17)
target_link_libraries(stt-worker PRIVATE whisper)
