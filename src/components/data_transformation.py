import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_pickle_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            categorical_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                                   'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                                   'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                                   'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                                   'spore-print-color', 'population', 'habitat']

            # create pipeline to impute missing data and encode the columns to numerical
            cat_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder())
            ]
            )

            preprocessor = ColumnTransformer([
                ("cat", cat_pipeline, categorical_columns)
            ])

            logging.info("Preprocessor file has been successfully created")
            return preprocessor

        except Exception as err:
            raise CustomException(sys, err)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation has been initiated")

            # read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # since it cant be recognized as NAN value i replaced it so that pipeline can understand it
            train_df.replace("?", np.nan, inplace=True)
            test_df.replace("?", np.nan, inplace=True)

            logging.info(f"Train dataframe head: \n {train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n {test_df.head().to_string()}")

            target_feature_col = "class"
            drop_column = "veil-type"  # since it has only one feature

            # split the data into input and target feature
            input_feature_train_df = train_df.drop([target_feature_col, drop_column], axis=1)
            target_feature_train = train_df[target_feature_col]
            input_feature_test_df = test_df.drop([target_feature_col], axis=1)
            target_feature_test = test_df[target_feature_col]
            logging.info("Input and Target feature has been successfully split")

            # transform the feature using preprocessor
            preprocessor_obj = self.get_data_transformation_object()
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test)]
            logging.info("Data Transformation has been successfully completed")

            save_object(self.data_transformation_config.preprocessor_pickle_path, preprocessor_obj)
            logging.info("Preprocessor picked was successfully saved")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_pickle_path)
        except Exception as err:
            raise CustomException(sys, err)