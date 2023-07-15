import logging
import sys
import os
import pandas as pd
from src.utils import load_object


class CustomData:
    def __init__(self, cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                 gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                 stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type,
                 spore_print_color, population, habitat):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat

    def get_data_as_dataframe(self):
        data_dict = {
            "cap-shape": [self.cap_shape],
            "cap-surface": [self.cap_surface],
            "cap-color": [self.cap_color],
            "bruises": [self.bruises],
            "odor": [self.odor],
            "gill-attachment": [self.gill_attachment],
            "gill-spacing": [self.gill_spacing],
            "gill-size": [self.gill_size],
            "gill-color": [self.gill_color],
            "stalk-shape": [self.stalk_shape],
            "stalk-root": [self.stalk_root],
            "stalk-surface-above-ring": [self.stalk_surface_above_ring],
            "stalk-surface-below-ring": [self.stalk_surface_below_ring],
            "stalk-color-above-ring": [self.stalk_color_above_ring],
            "stalk-color-below-ring": [self.stalk_color_below_ring],
            "veil-color": [self.veil_color],
            "ring-number": [self.ring_number],
            "ring-type": [self.ring_type],
            "spore-print-color": [self.spore_print_color],
            "population": [self.population],
            "habitat": [self.habitat]
        }

        df = pd.DataFrame(data_dict)
        return df


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        # Set the path for pickle file
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        model_path = os.path.join("artifacts", "best_model.pkl")
        # load the pickle file
        preprocessor_file = load_object(preprocessor_path)
        model_file = load_object(model_path)
        data = preprocessor_file.transform(data)
        logging.info("Data has been successfully transformed in prediction pipeline")
        # predict the classification and find its probability
        predicted = model_file.predict(data)
        probability = model_file.predict_proba(data)
        logging.info("Data has been successfully predicted in prediction pipeline")
        return predicted, probability
