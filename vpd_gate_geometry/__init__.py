"""Spectral / tensor geometry of VPD causal-importance gates.

This package provides a tiny analysis pipeline over the causal-importance
field g[layer, batch, token, component] produced by the VPD method
(Goodfire `param-decomp`). It is deliberately minimal: extraction,
flattening, same-position kernels, token residualization, lagged
kernels, and a small set of Goodfire-style plots.

The point of the package is to study the *gate field* as a first-class
object, not to reimplement VPD itself.
"""

__version__ = "0.1.0"
