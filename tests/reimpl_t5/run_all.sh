#!/bin/bash

# Run on CPUs
python -m unittest tests.reimpl_t5.test_weight_matching
python -m unittest tests.reimpl_t5.test_small_components
python -m unittest tests.reimpl_t5.test_large_components

# Run on GPUs
python -m unittest tests.reimpl_t5.test_integration
deepspeed tests/reimpl_t5/test_e2e.py
