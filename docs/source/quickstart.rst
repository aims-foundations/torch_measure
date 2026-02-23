Quick Start
===========

Fitting a Rasch Model
---------------------

.. code-block:: python

   import torch
   from torch_measure.models import Rasch

   # Create synthetic response data (20 subjects x 30 items)
   responses = torch.bernoulli(torch.full((20, 30), 0.5))

   # Fit the model
   model = Rasch(n_subjects=20, n_items=30)
   history = model.fit(responses, max_epochs=500)

   # Get estimated parameters
   print("Abilities:", model.ability)
   print("Difficulties:", model.difficulty)

Running Adaptive Testing
------------------------

.. code-block:: python

   from torch_measure.cat import AdaptiveTester

   # Use a fitted model
   tester = AdaptiveTester(model, strategy="fisher")

   # Simulate a test for one subject
   true_responses = torch.bernoulli(torch.full((30,), 0.6))
   result = tester.run(true_responses, budget=10)
   print("Estimated ability:", result["ability"])

Computing Psychometric Metrics
------------------------------

.. code-block:: python

   from torch_measure.metrics import cronbach_alpha, mokken_scalability

   alpha = cronbach_alpha(responses)
   print(f"Cronbach's alpha: {alpha:.3f}")

   scalability = mokken_scalability(responses)
   print(f"Mokken H: {scalability['H']:.3f}")
