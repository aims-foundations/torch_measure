Installation
============

Basic Install
-------------

.. code-block:: bash

   pip install torch-measure

Optional Dependencies
---------------------

.. code-block:: bash

   # Bayesian estimation (Pyro SVI)
   pip install torch-measure[bayesian]

   # Visualization (matplotlib, seaborn)
   pip install torch-measure[viz]

   # Data loaders (HuggingFace Hub)
   pip install torch-measure[data]

   # Everything
   pip install torch-measure[all]

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/aims-foundation/torch_measure.git
   cd torch_measure
   pip install -e ".[dev,test]"
