# pygrad

PyGrad is a deep learning framework built mainly for educational purposes.
The goal is to reimplment some features that are similar to other well-known
Python packages, like PyTorch and TensorFlow.

To get started with development, you'll need to install the following:

```bash
sudo apt install cmake
sudo apt install clang-format
```

Mak sure to use c++ 17 standard. This project uses [GoogleTests](https://github.com/google/googletest) for unit testing. To run them, use the following commands:

```bash
cmake -S . -B build
cmake --build build
cd build && ctest
```
