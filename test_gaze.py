from gaze import Gaze
from example.model import MnistModel

test_model = MnistModel()
gazer = Gaze(test_model.network)

print(gazer.model.keys())

