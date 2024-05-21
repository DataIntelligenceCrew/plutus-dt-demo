# PLUTUS

Code for the SIGMOD 2024 demo paper PLUTUS: Understanding data distribution tailoring for machine learning.

This is a human-in-the-loop model training proof of concept. Users can run machine learning models, find problematic slices, then enrich the dataset with additional tuples of problematic slices acquired efficiently from external data sources. 

## Usage

A task asks to train an initial model, then enrich the train set with additional data from other sources. For maximum flexibility, tasks our now defined as Python classes. See the `AbstractTask` class in `dt/task.py` for further documentation. For standard tasks that involve classification or regression on data that is already cleaned, helper functions are pre-defined
