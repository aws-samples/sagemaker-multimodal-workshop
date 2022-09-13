from __future__ import print_function
import json
import bisect



import os
import numpy as np
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
# import pandas as pd
import argparse
import logging

# from pudb import set_trace

# from sagemaker_training import environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def model_fn(self, model_dir):
    """
    Load the model for inference
    """
    logger.info("************* loading log *****************")
    print("************* loading *****************")
    
#     set_trace()
    model_path = os.path.join(model_dir, 'model/')
    
    # model from disk.
    loaded_predictor = MultiModalPredictor.load(model_path)
    
    return loaded_predictor


def predict_fn(self, input_data, model):
    """
    Apply model to the incoming request
    """
    
    logger.info('predicting')
    # scores = model.evaluate(test_data, metrics=["roc_auc"])
#     scores = str(model)
    
    return scores



def transform_fn(mod, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param mod: The super resolution model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    
    logger.info("************* transform log *****************")
    print("************* transform  *****************")
    input_data = json.loads(data)
    logger.info(input_data)
#     batch = namedtuple("Batch", ["data"])
#     mod.forward(batch([mx.nd.array(input_data)]))
#     return (
#         json.dumps(mod.get_outputs()[0][0][0].asnumpy().clip(0, 255).tolist()),
#         output_content_type,
#     )
    return {"reply": "reply"}