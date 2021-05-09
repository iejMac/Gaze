# Gaze
Visualize PyTorch Models

### Streaming:
Pull up a live visualization of model training or inference:
```
model = Model()
# Pass nn.Module and optimizer into the Gaze wrapper
gaze = Gaze(model.network, model.optimizer)

# Show how gradients are flowing through a layer during training
gaze.streamGradient("layer_name")

# Show how weights during training or inference
gaze.streamWeight("layer_name")

# Start using the model and Gaze will show you the weights and/or the gradients
if train:
  model.train()
elif test:
  model.test()
```

### Checking:
You might only want to peer into the model under specific conditions. For this we have checking:
```
from gaze import Gaze

class Model:
  def __init__(self):
    self.network = Network(...) # nn.Module
    self.optimizer = optim.SomePytorchOptimizer(...) 
    self.gaze = Gaze(self.network, self.optimizer)

  def train(epochs):
    for e in range(epochs):
      for i, data_batch in enumerate(data):
        self.optimizer.zero_grad()
        output = self.network(data_batch)
        loss = somePytorchLoss(output, labels[i])

        # Only check weights and gradients when loss is lower than some threshold:
        if loss.item() < threshold:
          self.gaze.checkWeight("layer_name")
          self.gaze.checkGradient("layer_name")

        loss.backward()
        self.optimizer.step()
```
