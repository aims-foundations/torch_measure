Installation
============

Basic Install
-------------

.. code-block:: bash

   pip install torch_measure

Optional Dependencies
---------------------

.. code-block:: bash

   # Bayesian estimation (Pyro SVI)
   pip install torch_measure[bayesian]

   # Visualization (matplotlib, seaborn)
   pip install torch_measure[viz]

   # Data loaders (HuggingFace Hub)
   pip install torch_measure[data]

   # Everything
   pip install torch_measure[all]

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/aims-foundations/torch_measure.git
   cd torch_measure
   pip install -e ".[dev,test]"
