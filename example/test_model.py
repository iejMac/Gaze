from model import MnistModel
import torch

model = MnistModel()
model.train(3)
model.test()
model.saveModel()
