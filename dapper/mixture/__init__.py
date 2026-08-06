"""Mixture planning: target percentages checked against measured capacities.

A mixture says what we *want*; the token manifest says what *exists*. They are
separate files because conflating them makes the only interesting question --
"is this mixture satisfiable?" -- unanswerable.
"""

from __future__ import annotations
