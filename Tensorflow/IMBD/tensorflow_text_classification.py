import bentoml
import pandas as pd
import tensorflow as tf
from bentoml import api
from bentoml.adapters import DataframeInput, DataframeOutput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from tensorflow import keras

""" 
use DataframeOutputAsymmetric instead of DataframeOutput 
if number of predicted rows should be more or less than in input data
"""
# from dataframe_output_asymmetric import DataframeOutputAsymmetric

REVIEW_CLASSES = ['negative', 'positive']

MAX_WORDS = 10000
word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def encode_review(text):
    words = text.split(' ')
    ids = [word_index["<START>"]]
    for w in words:
        v = word_index.get(w, word_index["<UNK>"])
        # >1000, signed as <UNseED>
        if v > MAX_WORDS:
            v = word_index["<UNUSED>"]
        ids.append(v)
    return ids


@bentoml.env(pip_dependencies=['tensorflow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class ImdbTensorflow(bentoml.BentoService):
    INPUT_COLUMNS = {"review": "str"}  # possible column types: int, float, str

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
        x = [encode_review(t) for t in df["review"]]
        x = keras.preprocessing.sequence.pad_sequences(x,
                                                       dtype="float32",
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
        y = self.artifacts.model(x)
        output = pd.DataFrame([REVIEW_CLASSES[c] for c in tf.argmax(y, axis=1).numpy().tolist()], columns=["Sentiment"])
        return output
