# Gaze
Visualize PyTorch Models

### Idea for how it will work:
```
model = Model()
gaze = Gaze(model)

# show how gradients are flowing throughout training or at time tn
gaze.checkGradients()

# show how weights are changing throughout training or inference or at time tn
gaze.checkWeights()

etc.

```

