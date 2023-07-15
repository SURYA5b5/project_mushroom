import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.utils import save_object
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from dataclasses import dataclass
import numpy as np
import statistics
from sklearn.model_selection import cross_val_score, KFold


@dataclass
class ModelTrainerConfig:
    model_pickle_path = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def cross_validation(self, name, model, train_arr, test_arr):
        # To check for accuracy for the different sample
        arr = np.concatenate((train_arr, test_arr), axis=0)
        y_arr = arr[:, -1]
        x_arr = arr[:, :-1]
        kf = KFold(n_splits=30)
        f1_macro = cross_val_score(model, x_arr, y_arr, cv=kf, scoring='f1_macro')
        logging.info(f"Model: {name}, Cv_F1 MACRO MEAN : {np.mean(f1_macro)}, Cv_STD DEV: {statistics.stdev(f1_macro)}")
        return np.mean(f1_macro)

    def initiate_model_training(self, train_arr, test_arr):
        try:
            # input and target feature split
            x_train, x_test, y_train, y_test = train_arr[:, :-1], test_arr[:, :-1], train_arr[:, -1], test_arr[:, -1]
            logging.info(f"train_test_shape: {x_train.shape, y_train.shape, x_test.shape, y_test.shape}")

            models = {
                "LogisticRegression": LogisticRegression(random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RandomForestClassifier":RandomForestClassifier(random_state=42),
                "GradientBoostingClassifier" :GradientBoostingClassifier(random_state=42)

   
            }

            f1_macro_list = []
            cross_f1_macro_list = []
            logging.info("Model training has been successfully initiated")

            # model training and evaluation
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = f1_score(y_test, y_pred, average='macro')
                f1_macro_list.append(score)
                logging.info(f"Model: {name}, F1 macro: {score}")
                cross_validation_score = self.cross_validation(name, model, train_arr, test_arr)
                cross_f1_macro_list.append(cross_validation_score)

            logging.info("Model training has been successfully completed")

            # finding the best model
            max_value = max(cross_f1_macro_list)
            max_value_index = cross_f1_macro_list.index(max_value)
            best_model_name = list(models.keys())[max_value_index]
            best_model = list(models.values())[max_value_index]
            logging.info(f"Best_model: {best_model_name}, F1 macro: {max_value}")

            save_object(self.model_trainer_config.model_pickle_path, best_model)
            logging.info("Best model was successfully saved")

            return f1_macro_list

        except Exception as err:
            raise CustomException(sys, err)
        
    