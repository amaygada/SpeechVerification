from train import train
from config import config
import numpy as np
import pandas as pd
import tensorflow as tf
from model import Embed, TransformerModel

model_config = config["model"]
epochs = model_config["epochs"]
batch_size = model_config["batch_size"]
train_cap = model_config["train_cap"]//batch_size
dev_cap = model_config["dev_cap"]//batch_size 

data_config = config["data_path"]
train_data_path = data_config["train"]
test_data_path = data_config["test"]
dev_data_path = data_config["dev"]

audio_data_config = config["audio_data_path"]
audio_train = audio_data_config["train"]
audio_test = audio_data_config["test"]
audio_dev = audio_data_config["dev"]

model_path = config["model_path"]
model_path_embed = model_path["embed"]
model_path_transformer = model_path["transformer"]

results_path = config["results"]["path"]

data_train = pd.read_csv(train_data_path)
data_eval = pd.read_csv(test_data_path)
data_dev = pd.read_csv(dev_data_path)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# def train(epochs, batch_size, train_data, dev_data, train_cap, dev_cap, model_path_embed, model_path_transformer, result_path, audio_train_path, audio_dev_path, embed, transformer):
embed = Embed()
transformer = TransformerModel()

embed, transformer = train(epochs, batch_size, data_train, data_dev, train_cap, dev_cap, model_path_embed, model_path_transformer, results_path, audio_train, audio_dev, embed, transformer)