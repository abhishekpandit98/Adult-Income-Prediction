import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os
from src.mlproject.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = ['age', 'education-num', 'hours-per-week']
            categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'country']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))])

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numeric columns: {numeric_features}")

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", num_pipeline, numeric_features),
                ("categorical_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = train_path
            test_df = test_path
            logging.info("Reading train and test file")

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'salary'
            input_features_train_df = train_df.drop(target_column, axis=1)
            target_features_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(target_column, axis=1)
            target_features_test_df = test_df[target_column]

            logging.info("Applying preprocessing on train and test dataframe")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            print("Shapes:")
            print("input_features_train_arr shape:", input_features_train_arr.shape)
            print("input_features_test_arr shape:", input_features_test_arr.shape)
            print("target_features_train_df shape:", target_features_train_df.shape)
            print("target_features_test_df shape:", target_features_test_df.shape)

            train_arr = np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
