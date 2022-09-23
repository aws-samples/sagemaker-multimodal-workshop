import os
import numpy as np
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import argparse
import logging
import glob
import tarfile
import time
import pathlib
import json




logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

label_col = 'AdoptionSpeed'
image_col = 'Images'

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="/opt/ml/processing")
    args, _ = parser.parse_known_args()
    return args


def load_data(file_list: list):
    # Define columns to use
    use_cols = []
    # Concat input files
    dfs = []
    for file in file_list:
        if len(use_cols)==0:
            dfs.append(pd.read_csv(file))
        else:
            dfs.append(pd.read_csv(file, usecols=use_cols))    
    return pd.concat(dfs, ignore_index=True)

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

def main(base_dir: str, args: argparse.Namespace):
    # Input test files
    input_dir = os.path.join(base_dir, "input/test")
    test_file_list = glob.glob(f"{input_dir}/*.csv")
    logger.info(f"Input file list: {test_file_list}")
        
    if len(test_file_list) == 0:
        raise Exception(f"No input files found in {input_dir}")

    # Input model
    model_dir = os.path.join(base_dir, "input/model")
    model_file = glob.glob(f"{model_dir}/*.tar.gz")
    logger.info(f"Model file: {model_file}")
    if not os.path.exists(model_dir):
        raise Exception(f"model file does not exist")
        
    model_path = f"{model_dir}/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_dir)
    

    # load data into dataframes
    test_data = load_data(test_file_list)
    test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=input_dir))
    
    for file in os.listdir(model_dir):
        logger.info(file)
    
    logger.info(" ** Loading model from file. **")
    loaded_predictor = MultiModalPredictor.load(model_dir)
    predictions = loaded_predictor.predict(test_data.drop(columns=label_col))
    print(predictions[:5])
    probas = loaded_predictor.predict_proba(test_data.drop(columns=label_col))
    print(probas[:5])
    
    # Write results to local file
    logger.info(" ** Writing prediction to file. **")
    predictions.to_json('/opt/ml/processing/output/inference_result/result.json')

    
    
    return

if __name__ == "__main__":
    logger.info(" ** Starting preprocessing. **")
    args = parse_args()
    base_dir = args.base_dir
    main(base_dir, args)
    logger.info("Done")