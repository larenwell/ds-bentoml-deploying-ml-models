# bento_service.py

# Import all necessary libraries from bentoml
import pandas as pd # You need to define pandas library
from sklearn.preprocessing import MinMaxScaler
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, DataframeOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact


@env(infer_pip_packages=True)
@artifacts([PickleArtifact('meta'),SklearnModelArtifact('model')])  # artifacts need to be decleared here
class IrisClassifier(BentoService):
    """
    Sample Sklearn Iris Flower Classification Model Wraper
    """
    
    INPUT_COLUMNS = {"petal_width": "float", "sepal_length": "float", "petal_length": "float", "sepal_width": "float"}
    
    # Scaler object 
    scaler_= MinMaxScaler()
    
    def pre_processing(self, df):
        """
        Scaling data using MixMaxScaler function
        :param: DataFrame object
        :return DataFrame object
        """
        return pd.DataFrame(self.scaler_.fit_transform(df.values), columns = self.INPUT_COLUMNS)
        
        
    def post_processing(self, predictions):
        """
        Convert output label token into target names
        :param predictions: list of int
        :return: Json
        """
        return pd.DataFrame([self.artifacts.meta["target_mapping"][i] for i in predictions],columns=["species"])

    @api(
        input=DataframeInput(
            orient="records",
            columns=INPUT_COLUMNS,
            dtype=INPUT_COLUMNS # possible column types: int, float, str
        ),
        batch=True,
        output=DataframeOutput(output_orient='records')
        )
    def predict(self, df: pd.DataFrame):
        """
        Prediction API method
        :param parsed_json_list: list of Json
        :return: Json
        """
        # -----!!!!!------
        # An inference API named `predict` with Dataframe input adapter, which defines
        # how HTTP requests or CSV files get converted to a pandas Dataframe object as the
        # inference API function input
        
        df_scaled = self.pre_processing(df)
        predictions = self.artifacts.model.predict(df_scaled)
        return self.post_processing(predictions)