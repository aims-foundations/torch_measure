Annotation
==========

Tools for annotating benchmark items with demand vectors using the
18-dimension ADeLe rubric system from the Nature 2026 paper.

:class:`DemandAnnotator` accepts any client that implements
``generate(prompt: str) -> tuple[str, str]``, so any LLM provider
(OpenAI, Anthropic, Azure, etc.) can be used in place of :class:`GeminiClient`
by wrapping it in a class with that single method.

.. automodule:: torch_measure.annotation
   :members:

Core Classes
------------

.. autoclass:: torch_measure.annotation.DemandAnnotator
   :members:
   :undoc-members:

.. autoclass:: torch_measure.annotation.GeminiClient
   :members:
   :undoc-members:

.. autoclass:: torch_measure.annotation.RubricsCatalog
   :members:
   :undoc-members:

.. autoclass:: torch_measure.annotation.AnnotationCache
   :members:
   :undoc-members:

Data Types
----------

.. autoclass:: torch_measure.annotation.AnnotationJob
   :members:
   :undoc-members:
