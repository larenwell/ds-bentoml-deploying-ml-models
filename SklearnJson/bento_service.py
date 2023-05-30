"""
BentoML service for Iris flower classification model.

Author: @columbia, @encora
Created: 5/6/2023

This file defines a BentoML service for a Sklearn model that classifies Iris flowers based on their petal and sepal dimensions. The service is wrapped using the IrisClassifier class, which includes a pre_processing method for scaling input data and a post_processing method for converting model predictions into human-readable labels. The service can be invoked using the predict method, which expects a JSON object with petal and sepal dimension data and returns a JSON object with the predicted flower species.
"""

"""
Define and instantiate the bentoml libraries
"""
import pandas as pd # You need to define pandas library
from sklearn.preprocessing import MinMaxScaler
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact


@env(infer_pip_packages=True)
@artifacts([PickleArtifact('meta'),SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    """
    Sample Sklearn Iris Flower Classification Model Wraper
    """
    INPUT_COLUMNS = {"petal_width": "float", "sepal_length": "float", "petal_length": "float", "sepal_width": "float"}

    # Scaler object 
    scaler_= MinMaxScaler()
    
    def pre_processing(self, input_data):
        """
        Scaling data using MixMaxScaler function
        :param input_data: dict
        :return DataFrame object
        """
        #print(input_data)
        df = pd.DataFrame(input_data,columns=self.INPUT_COLUMNS)
        #print(df)
        return pd.DataFrame(self.scaler_.fit_transform(df.values), columns = df.columns)
        
    def post_processing(self, prediction):
        """
        Convert output label token into target names
        :param predictions: array of int
        :return: dict
        """

        return {"species": self.artifacts.meta["target_mapping"][prediction]}

    @api(
        input=JsonInput(), #batch=True
        output=JsonOutput()
        )
    
    def predict(self, input_data:dict):
        """
        Prediction API method
        :param input_data: dict
        :return: dict
        """

        df_scaled = self.pre_processing(input_data)
        predictions = self.artifacts.model.predict(df_scaled)

        results = []
        for prediction in predictions:
            result = self.post_processing(prediction)
            results.append(result["species"])
        return {"predictions": results}
    

    

    