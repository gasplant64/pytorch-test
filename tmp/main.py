##Test pytorch##

import torch
import torch.nn as nn
import torch.optim as opt
import torch.functional as func
import numpy as np

from module import Normal
from module import Setmodel
from data import gen_num
#


x_train , y_train  = gen_num(length  = 250 , split = 5)
x_valid , y_valid  = gen_num(length  = 50 , split = 5)

## Classify 5*integer  number
Model =Normal(input_n = 1,hidden_n = 10,output_n = 2)



Fullmodel = Setmodel(model = Model , xtrain  = x_train , ytrain = y_train , xvalid = x_valid , yvalid = y_valid )
Fullmodel.show()
Fullmodel.train(epoch = 200 , bs = 30 , lr = 0.001)
Fullmodel.show()
Fullmodel.save('Trained')
