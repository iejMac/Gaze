import sys
import torch

class Gaze:
  def __init__(self, model, optimizer):
    self.raw_model = model
    self.optimizer = optimizer

  def checkWeight(self, key):
    observed = self.raw_model.state_dict()[key]
    # TODO: better way to round (very arbitrary rn)
    print(observed.numpy().round(2))
  def checkGradient(self, key):
    grad = (getattr(self.raw_model, key)).weight.grad
    print(grad)

  def streamWeight(self, key):
    old_forward = self.raw_model.forward
    def new_forward(x):
      self.checkWeight(key)
      return old_forward(x)
    self.raw_model.forward = new_forward
  def streamGradient(self, key):
    old_step = self.optimizer.step
    def new_step():
      old_step()
      self.checkGradient(key)
    self.optimizer.step = new_step
