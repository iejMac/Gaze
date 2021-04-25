# Gaze
Visualize PyTorch Models

### Idea:
```
model = Model()
gaze = Gaze(model)

# show how gradients are flowing throughout training or at time tn
if stream:
	gaze.streamGradients()
gaze.checkGradients(tn)

# show how weights are changing throughout training or inference or at time tn
if stream:
	gaze.streamWeights()
gaze.checkWeights(tn)
```
