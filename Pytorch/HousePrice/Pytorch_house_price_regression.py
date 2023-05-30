import bentoml
import pandas as pd
import torch
from bentoml import api
from bentoml.adapters import DataframeInput, DataframeOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from torch.autograd import Variable

""" 
use DataframeOutputAsymmetric instead of DataframeOutput 
if number of predicted rows should be more or less than in input data
"""
# from dataframe_output_asymmetric import DataframeOutputAsymmetric


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('net')])
class HousePricePytorch(bentoml.BentoService):
    INPUT_COLUMNS = {"CRIM": "float", "ZN": "float", "INDUS": "float", "CHAS": "float", "NOX": "float", "RM": "float",
                     "AGE": "float", "DIS": "float", "RAD": "float", "TAX": "float", "PTRATIO": "float", "B": "float",
                     "LSTAT": "float"}  # possible column types: int, float, str

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
        results = self.artifacts.net(Variable(torch.FloatTensor(df[self.INPUT_COLUMNS].values))).detach().numpy()
        return pd.DataFrame(results, columns=["price"])
