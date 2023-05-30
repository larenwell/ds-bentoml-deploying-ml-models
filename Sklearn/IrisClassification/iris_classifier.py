import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput, DataframeInput, DataframeOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact

""" 
use DataframeOutputAsymmetric instead of DataframeOutput 
if number of predicted rows should be more or less than in input data
"""
# from dataframe_output_asymmetric import DataframeOutputAsymmetric


@env(infer_pip_packages=True)
@artifacts([PickleArtifact('meta'), SklearnModelArtifact('model')])  # artifacts need to be decleared here
class IrisClassifier(BentoService):
    """
    Sample Sklearn Iris Flower Classification Model Wraper
    """

    INPUT_COLUMNS = {"petal_width": "float", "sepal_length": "float", "petal_length": "float", "sepal_width": "float"}
    # possible column types: int, float, str

    def pre_processing(self, parsed_json_list):
        """
        convert input json into 2D list in the right order for model prediction
        :param parsed_json_list: list of Json
        :return: 2D list
        """
        feature_names = self.artifacts.meta["feature_names"]
        return [[j[key] for key in feature_names] for j in parsed_json_list]

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
            dtype=INPUT_COLUMNS
        ),
        batch=True,
        output=DataframeOutput(output_orient='records')
        # -----!!!!!------
        # use DataframeOutputAsymmetric instead of DataframeOutput
        # if number of predicted rows should be more or less than in input data
        # output = DataframeOutputAsymmetric(output_orient='records')
    )
    def predict(self, df):
        """
        Prediction API method
        :param parsed_json_list: list of Json
        :return: Json
        """
        predictions = self.artifacts.model.predict(df)
        return self.post_processing(predictions)

    @api(input=JsonInput(), output=JsonOutput())
    def meta(self, NULL):
        """
        API to display meta data
        :param NULL:
        :return:
        """
        return [self.artifacts.meta]
