import os
import numpy as np
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import argparse
import logging

from sagemaker_training import environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

def _train(args):

    train_data = pd.read_csv(f'{args.data_dir}/train.csv', index_col=0)
    
    label_col = 'AdoptionSpeed'
    image_col = 'Images'


    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=args.data_dir))

    predictor = MultiModalPredictor(label=label_col)
    predictor.fit(
        train_data=train_data,
        time_limit=120, # seconds
        save_path=args.model_dir,
    )


    logger.info("Saving the model.")
    predictor.save(args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    _train(parser.parse_args())