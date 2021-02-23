##Test pytorch##

import torch
import torch.nn as nn
import torch.optim as opt
import torch.functional as func
import numpy as np

from module import normal
from module import Setmodel
from data import gen_num
#


xt , yt  = gen_num(length  = 250 , split = 5)
xv , yv  = gen_num(length  = 50 , split = 5)

## Classify 5*integer  number
Model = normal(inputs = 1,nodes = 10,outputs = 2)



Fullmodel = Setmodel(model = Model , xtrain  = xt , ytrain = yt , xvalid = xv , yvalid = yv )
Fullmodel.show()
Fullmodel.train(epoch = 200 , bs = 30 , lr = 0.001)
Fullmodel.show()
Fullmodel.save('Trained')
