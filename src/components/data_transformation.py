import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_transformer_obj(self):
        try:
            num_cols = ["age", "bmi", "children"]
            cat_cols = ["sex", "region"]

            numeric_transformer = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            logging.info('Numerical columns scaled')
            logging.info('Categorical columns encoded')
  
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_cols),
                    ("cat", categorical_transformer, cat_cols),
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("The train and test data reading complete")
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_obj()
            target_feature = 'charges'
            num_cols = ["age", "bmi", "children"]
            
            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]
            
            logging.info("Applying preprocessing object on train and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e, sys)
