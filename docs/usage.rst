Usage
=============

Setup
-------

mlf-core based mlflow projects require either Conda or Docker to be installed.
The usage of Docker is highly preferred, since it ensures that system-intelligence can fetch all required and accessible hardware.
This cannot be guaranteed for macOS let alone Windows environments.

Conda
+++++++

There is no further setup required besides having Conda installed and CUDA configured for GPU support.
mlflow will create a new environment for every run.

Docker
++++++++

If you use Docker you need to build the Docker container, whose name is specified in the MLproject file (e.g. mlfcore/pytorch:1.0.0).
This is sufficient to train on the CPU. If you want to train using the GPU you need to have the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ installed.

The ``data`` folder is by default mounted to ``/data`` inside the docker container.

Training
-----------

Training on the CPU
+++++++++++++++++++++++

Set your desired environment in the MLproject file. You need to disable CUDA to train on the CPU!
Start training using ``mlflow run . -P cuda=False -P training-data='/data/train.tsv' -P test-data='/data/test.tsv'``.

Training using GPUs
+++++++++++++++++++++++

Conda environments will automatically use the GPU if available.
Docker requires the accessible GPUs to be passed as runtime parameters.
To train using all gpus run ``mlflow run . -P training-data='/data/train.tsv' -P test-data='/data/test.tsv' -A gpus=all``.

Parameters
+++++++++++++++

- training-data               Path to the training data csv file                          ['train.csv': string]
- test-data                   Path to the test data csv file                              ['test.csv':  string]
- cuda:                       Whether to train with CUDA support (=GPU)                   ['True':      string]
- epochs:                     Number of epochs to train                                   [25:             int]
- general-seed:               Python, Random, Numpy seed                                  [0:              int]
- xgboost-seed:               XGBoost specific seed                                       [0:              int]
- single-precision-histogram  Whether to enable `single precision for histogram building <https://xgboost.readthedocs.io/en/latest/parameter.html#additional-parameters-for-hist-and-gpu-hist-tree-method>`_ ['True': string]
