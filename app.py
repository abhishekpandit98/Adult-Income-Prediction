from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
import os
import pandas as pd
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
 
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        ##loading train and test data to perform data transformation
        train_data = pd.read_csv(os.path.join('artifacts', 'train_df.csv'))
        test_data = pd.read_csv(os.path.join('artifacts', 'test_df.csv'))
        logging.info("loading train and test data done")

        ##performing data transformation
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info("performing data transformation done")

    except Exception as e:
        logging.error("Custom Exception")
        raise CustomException(e, sys)