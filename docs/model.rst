Model
======

Overview
~~~~~~~~~~

The hereby trained model classifies samples generated from gene expression data into cancerous or benign.

Training and test data
~~~~~~~~~~~~~~~~~~~~~~~~

Patients with cancer and without cancer were sequenced (RNA-Seq). All samples of the patients were assigned 'cancerous' or 'benign'.
The RNA-Seq experiments generated reads, which are commonly associated with expression values per gene.
Furthermore, all replicates were merged into single samples resulting in median gene expression values per gene.
These expression values were normalized into transcripts per million (TPM) values.
Next, all genes were subject to human pathway analysis. Any genes not present in any pathway was discarded.
Finally, the whole dataset was split into 80% training and 20% test data.
lcep was trained with the training data and evaluated using the test data.

Model details
~~~~~~~~~~~~~~

The model is based on `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_.
Training was conducted using a single GPU (NVIDIA 1050M), which is also reported in the system-intelligence report.
Hence, ``gpu_hist`` is the training algorithm of choice.

Evaluation
~~~~~~~~~~~~~

The model was evaluated on 20% of unseen test data. The reported root mean squared error origins from the test data.
The full training history is viewable by running the mlflow user interface inside the root directory of this project:
``mlflow ui``.

Hyperparameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

No sophisticated Hyperparameter selection was conducted due to time constraints.

1. ``single-precision-histogram`` was enabled for faster training.
2. ``subsample`` was set to 0.5 for arbitrary reasons.
3. ``colsample_bytree`` was set to 0.5 for arbitrary reasons.
4. ``colsample_bylevel`` was set to 0.5 for arbitrary reasons.
