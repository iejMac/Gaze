import sys
import torch

class Gaze:
	def __init__(self, model):
		self.raw_model = model

	def streamWeight(self, key):
		old_forward = self.raw_model.forward

		def new_forward(x):
			observed = self.raw_model.state_dict()[key]
			print("===================")
			print(observed.numpy().round(2))
			return old_forward(x)

		self.raw_model.forward = new_forward

# Need to set this up so that whenever the model calls forward, the thing is printed etc.

