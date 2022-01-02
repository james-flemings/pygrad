# pygrad

PyGrad is a deep learning framework built mainly for educational purposes.
The goal is to reimplment some features that are similar to other well-known
Python packages, like PyTorch and TensorFlow. 

This project uses [GoogleTests](https://github.com/google/googletest) for unit testing. To run them, use the following commands:
```bash
cmake -S . -B build
cmake --build build
cd build && ctest
```