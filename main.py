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


xtrain , ytrain  = gen_num(length  = 250 , split = 5)
xvalid , yvalid  = gen_num(length  = 50 , split = 5)


Model = normal(inputs = 1,nodes = 10,outputs = 2)
