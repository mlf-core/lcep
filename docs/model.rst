Model
======

Overview
~~~~~~~~~~

The hereby trained model classifies samples generated from gene expression data into cancerous or healthy.

Training and test data
~~~~~~~~~~~~~~~~~~~~~~~~

Patients with cancer and without cancer were sequenced (RNA-Seq). All samples of the patients were assigned 'cancerous' or 'healthy'.
The RNA-Seq experiments generated reads, which are commonly associated with expression values per gene.
These expression values were normalized into transcripts per million (TPM) values.
Next, all genes were subject to pathway analysis. Any genes not present in any cancer associated pathway were discarded.
Finally, the whole dataset was split into 75% training and 25% test data.
lcep was trained with the training data and evaluated using the test data.

Model details
~~~~~~~~~~~~~~

The model is based on `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_.
Training was conducted using a single GPU . Hence, ``gpu_hist`` is the training algorithm of choice.

Evaluation
~~~~~~~~~~~~~

The model was evaluated on 20% of unseen test data. The reported root mean squared error origins from the test data.
The full training history is viewable by running the mlflow user interface inside the root directory of this project:
``mlflow ui``.

Hyperparameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hyperparameters of this model were selected using a grid search approach.

1. ``single-precision-histogram`` was enabled for faster training
2. ``subsample`` was set to 0.7
3. ``colsample_bytree`` was set to 0.6
4. ``learning_rate`` was set to 0.2
5. ``max_depth`` was set to 3
6. ``min_child_weight`` was set to 1
7. ``eval_metric`` was set to ``logloss``
8. ``objective`` was set to ``binary:logistic``
