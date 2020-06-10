import torch
import torch.nn as nn
import numpy as np

def ShuffledArray(G,C):
  a = []
  count = 1
  if(G == 1):
    sG = C
  elif(G == C):
    sG = 1
  else:
    sG = G
  for i in range(1,C+1):
    a.append(count)
    if(i%(C//sG) == 0):
      count = count + 1
  a = np.array(a)
  np.random.shuffle(a)
  return a

class GroupShuffleNorm2d(nn.Module):
    def __init__(self, in_features, in_groups, epsilon=1e-5):
        super(GroupShuffleNorm2d, self).__init__()
        self.in_groups = in_groups
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1,in_features,1,1))
        self.beta = nn.Parameter(torch.zeros(1,in_features,1,1))
    def forward(self, x):
        samples,channels,dim1,dim2 = x.shape
        s_array = ShuffledArray(self.in_groups,channels)
        attach_mean = {}
        index = 0
        for j in range(channels):
          if s_array[index] not in attach_mean:
            attach_mean[s_array[index]] = [index]
          else:
            attach_mean[s_array[index]].append(index)
          index +=1
        for i in attach_mean.keys():
          mean_is = torch.mean(x[:,attach_mean[i],:,:,],axis = (1,-1,2)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
          var_is = torch.var(x[:,attach_mean[i],:,:,],axis = (1,-1,2)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
          x[:,attach_mean[i],:,:,] = (x[:,attach_mean[i],:,:,] - mean_is) / (var_is+self.epsilon).sqrt()
        return x * self.gamma + self.beta