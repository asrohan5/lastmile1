import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor,\
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Training and Test input data")
            Xtrain, ytrain, Xtest, ytest = (train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1],)
            
            models = {
                "RandomForest" : RandomForestRegressor(),
                "DecisionTree" : DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNNRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False),
                "AdaBoostRegressor":AdaBoostRegressor(),}
            

            model_report: dict=evaluate_models(Xtrain=Xtrain, Xtest = Xtest, 
                                              ytrain=ytrain, ytest = ytest, 
                                              models = models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model =  models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best Model Found")
            
            logging.info(f"Best FOund model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(Xtest)
            r_square = r2_score(ytest, predicted)
            return r_square


        except Exception as e:
            raise CustomException(e,sys)