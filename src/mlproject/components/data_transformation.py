import numpy as np
import pandas as pd 
import dataclasses import dataclass 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(Self):
        self.data_trasformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numeric_features = x.select_dtypes(include='number').columns.tolist()
            categorical_features = x.select_dtypes(exclude='number').columns.tolist()

        num_pipeline=(steps=[
            ("imputer",SimpleImputer(strategy='median')),
            ("scaler",StandardScaler())
        ])
        cat_pipeline=(steps=[
            ("imputer",SimpleImputer(strategy='most_frequent')),
            ("ohe",OneHotEncoder()),
            ("scaler",StandardScaler(with mean=False))
        ])

        logging.info(f"Catogrical columns:{categorical_features}")
        logging.info(f"Catogrical columns:{numeric_features}")

        preprocessor=ColumnTransformer([
            ("mumericalpipeline",num_pipeline,numeric_features),
            ("catogorial[ipeline]",cat_pipeline,categorical_features)
        ])
        return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def  initiate 