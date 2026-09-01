"""Batch monitoring for the deployed model.

``reference``  build the distribution snapshot the live batches are compared to
``drift``      PSI / KS primitives
``batch``      score one daily batch against the reference -> a metrics record
"""
