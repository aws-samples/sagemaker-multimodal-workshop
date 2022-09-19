'''
This is a realtime endpoint script.
It is servicing as a placeholder for now.
Until official release, you can only use this example as a reference for loading models and invocation handler.

'''
from autogluon.multimodal import MultiModalPredictor


import os
import json
from io import StringIO
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    print("********* Loading mode ****************")
    model = MultiModalPredictor.load(model_dir)
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
# def predict_fn(self, data, model):
    print("********* TRANS mode ****************")
#     pred = model.predict(data)
#     pred_proba = model.predict_proba(data)
#     prediction = pd.concat([pred, pred_proba], axis=1)

    return "result"