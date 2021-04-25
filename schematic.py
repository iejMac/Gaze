import numpy as np

class TestModel:
  def __init__(self, kern_size):
    self.filter = np.random.rand(kern_size, kern_size)
  def forward(self, x):
    self.filter = self.filter @ x
    return self.filter

class TestGaze:
  def __init__(self, subject):

    old_forward = subject.forward
    
    def new_forward(x):
      print("Look at this important weight! : " + str(subject.filter[1][1]))
      return old_forward(x)
      
    subject.forward = new_forward

epochs = 10

model = TestModel(3)
gazer = TestGaze(model)

for i in range(epochs):
  x = np.random.rand(3, 3)
  print(i)
  print(model.forward(x))
