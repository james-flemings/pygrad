cmake_minimum_required(VERSION 3.14)
project(pygrad)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(include)

add_executable(
  Neuron_test
  tests/Neuron_test.cpp
  tests/DenseLayer_test.cpp
  tests/Model_test.cpp
  src/Neuron.cpp
  src/Initializer.cpp
  src/Layer.cpp
  src/DenseLayer.cpp
  src/Model.cpp
)
target_link_libraries(
  Neuron_test
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(Neuron_test)