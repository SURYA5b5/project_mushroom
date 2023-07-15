import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pymongo


@dataclass
class LoadDataConfig:
    raw_path = os.path.join("artifacts", "raw.csv")
    input_data_path = os.path.join("notebooks/data", "mushrooms.csv")


class LoadData:
    def __init__(self):
        self.load_data_config = LoadDataConfig()
        self.db = None
        self.collection = None

    def create_database(self):
        try:
            # connects the mongodb_database
            client = pymongo.MongoClient(
                f"mongodb+srv://suryateja:suryateja@cluster0.vulgqrg.mongodb.net/")
            logging.info("Successfully connected to the MongoDB database.")

            # Create the database and collection
            self.db = client["predict_mushrooms"]
            self.collection = self.db["mushrooms"]
            # Read the data from the source csv file
            df = pd.read_csv(self.load_data_config.input_data_path)
            dict_li = df.to_dict('list')
            # Insert data into the collection
            self.collection.insert_one(dict_li)
            logging.info("Created the Mushroom Database successfully.")
        except Exception as err:
            raise CustomException(sys, err)

    def read_database(self):
        try:
            # Read the data from database and create the data as dataframe
            document = self.collection.find_one()
            df = pd.DataFrame(document)
            df.drop(["_id"], axis=1, inplace=True)
            logging.info("Successfully retrieved data from MongoDB.")

            # Save the raw data in the artifacts folder
            os.makedirs(os.path.dirname(self.load_data_config.raw_path), exist_ok=True)
            df.to_csv(self.load_data_config.raw_path)
            logging.info("Raw data has been saved successfully.")
        except Exception as err:
            raise CustomException(sys, err)

if __name__ == "__main__":
    load_data=LoadData()
    load_data.create_database()
    load_data.read_database()
