import sys
import torch

class Gaze:
  def __init__(self, model):
    self.raw_model = model

  def streamWeight(self, key):
    old_forward = self.raw_model.forward
    def new_forward(x):
      observed = self.raw_model.state_dict()[key]
      print(observed.numpy().round(2))
      return old_forward(x)
    self.raw_model.forward = new_forward

  # TODO: grad gets populated on loss.backward(), make this work.
  def streamGradients(self, key):
    old_forward = self.raw_model.forward
    def new_forward(x):
      f_pass = old_forward(x)

      grad = (getattr(self.raw_model, key)).weight.grad
      print(grad)

      return f_pass
    self.raw_model.forward = new_forward
