# Hands' On TensorFlow
Introduction to TensorFlow

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

## Start Working
The project is organized as follow:
* The tutorials/exercices module is your working module
* The tutorials/solutions module contains the solutions of the exercices

To complete the exercices, follow the instructions in the presentation.
