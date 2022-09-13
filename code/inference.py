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
    print("********* TRANS mode ****************")
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)

    return prediction.to_json(), output_content_type