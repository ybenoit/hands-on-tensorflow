# Hands' On TensorFlow
Introduction to Neural Networks and TensorFlow with guided exercises.

## Setup

#### Anaconda
The project is based on a Python 2 Anaconda distribution.
Download it from the main website if needed.

#### Create an Anaconda environment
It is recommended to create a separate environment to work with TensorFlow to avoid any version problem in your main Python environment.
Source the environment once created.

```
conda create --name=tensorflow_env python=2.7
source activate tensorflow_env
```

### Install TensorFlow
Before downloading TensorFlow, make sure you don't have a installation of the protobuf library, which has some version problems with TensorFlow.
Once done, you can download TensorFlow.

```
pip uninstall protobuf
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl
pip install --upgrade $TF_BINARY_URL
```

### Clone the repository
```
git clone https://github.com/ybenoit/hands-on-tensorflow.git
cd hands-on-tensorflow
```

### Run tests
```
cd hands-on-tensorflow
py.test hands_on_tensorflow/solutions/mnist_1_softmax_regression/tests/
```

### Configure your IDE
To be able to work with TensorFlow in an IDE like PyCharm, you must specify the Python Interpreter corresponding to your TensorFlow conda environment.
* Open PyCharm
* PyCharm -> Preferences -> Project Interpreter
* Select the interpreter corresponding to your TensorFlow anaconda environment (Ex: ~/anaconda/envs/tensorflow/bin/python)

## Start Working
The project is organized as follow:
* The tutorials/exercices module is your working module
* The tutorials/solutions module contains the solutions of the exercices

To complete the exercices, follow the instructions in the presentation (prez directory).
