import torch

class Gaze:
	def __init__(self, model):
		self.raw_model = model
		self.parseModel(model)
		
	def parseModel(self, model):
		self.model = self.raw_model.state_dict()

# Need to set this up so that whenever the model calls forward, the thing is printed etc.

