import torch

test = torch.load("example/mnist_model")

for key in test.keys():
	print(key, test[key].shape)
	inp = input()
	if inp == "y":
		print(test[key])

