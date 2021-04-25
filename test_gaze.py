from gaze import Gaze
from example.model import MnistModel

test_model = MnistModel()
gazer = Gaze(test_model.network)
gazer.streamWeight("conv2.weight")
# gazer.streamGradients("conv2")

test_model.train(1)
test_model.test()
