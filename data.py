## make data set ##
import numpy as np
import numpy.random as rd
import torch
## Data generate Randum number of 0~100 ##
## Classify resultant of integer        ##


def gen_num(length , split , num = 100):
  xtmp = []
  for _ in range(length):
    xtmp.append(rd.randint(1,num))
  ytmp = []
  for i in xtmp: 
    if i % split == 0: ytmp.append(1)
    else: ytmp.append(0)
  xdata = torch.tensor(xtmp , dtype = torch.float)
  ydata = torch.tensor(ytmp)
  return xdata , ydata




if __name__ == '__main__':
  print(gen_num(length = 100,split = 4))
  
  
