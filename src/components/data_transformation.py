import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_col = ['writing_score', 'reading_score']
            cat_col = ['gender',
                        'race_ethnicity',
                        'parental_level_of_education',
                        'lunch',
                        'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler(with_mean=False))
                ])
            
            logging.info('Num Cols: {num_cols}')
            #logging.info('Numerical Columns scaling completed')
            
            cat_pipeline = Pipeline(steps =[
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('OneHotEncoding', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
                ])
            
            logging.info('Cat Col : {cat_cols}')
            #logging.info('Categorical Columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_col),
                    ('cat_pipeline', cat_pipeline, cat_col)
                ]
                ) 
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test data path completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_columns_name = 'math_score'

            num_col = ['writing_score', 'reading_score']
            '''
            cat_col = ['gender',
                        'race_ethnicity',
                        'parental_level_of_education',
                        'lunch',
                        'test_preparetion_course']
            '''
            input_feature_train_df = train_df.drop(columns = [target_columns_name], axis = 1)
            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(columns = [target_columns_name], axis = 1)
            target_feature_test_df = test_df[target_columns_name]

            logging.info(f"applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing Object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)