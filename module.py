### Pytorch module ###
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
import math



### DEFINE 1 Layer NN ##
class normal(nn.Module):
  def __init__(self, inputs , node , out ):
    super().__init__()
    self.inputs = inputs ; self.node = node
    self.lay1 = nn.Linear(inputs, node)
    self.lay2 = nn.Linear(node, out)
  def forward(self, tmp):
    tmp = tmp.view(-1,self.inputs)
    tmp = self.lay1(tmp) 
    tmp = func.relu(tmp)
    tmp = self.lay2(tmp)
    return tmp
  def accuracy(self, xb , yb):
    #print(self.forward(xb))
    pred = torch.argmax( self.forward(xb) , dim = 1)
    return (pred == yb).float().mean()



#elen(self.xtrain) - 1) // bs + 1# Classcco traning model ##
class Setmodel:
  def __init__(self, model , xtrain , ytrain , xvalid = False , yvalid = False):
    self.model = model
    self.xtrain = xtrain
    self.ytrain = ytrain
    self.xvalid = xvalid
    self.yvalid = yvalid
    #LOSS function##
    self.loss_func = nn.CrossEntropyLoss()
  def train(self , epoch  , bs , lr):
    num = (len(self.xtrain) - 1) // bs + 1
    for epoch in range(epoch):
         tloss = 0
         for i in range(num):
            start_i = i * bs
            end_i = start_i + bs
            xb = self.xtrain[start_i:end_i]
            yb = self.ytrain[start_i:end_i]
            pred = self.model(xb)
            loss = self.loss_func(pred, yb)
            tloss += loss
            
            loss.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    p -= p.grad * lr
                self.model.zero_grad()
         print('Epoch {0} Train Loss : {1}'.format( epoch+1 ,tloss/num) )
         if self.xvalid and self.yvalid:
           numv = (len(self.xvalid) - 1) // bs + 1
           vloss = 0
           for j in range(num):
             start_j = j * bs
             end_j = start_j + bs
             xv = self.xvalid[start_j:end_j]
             yv = self.yvalid[start_j:end_j]
             pred = self.model(xv)
             loss = self.loss_func(pred, yv)
             vloss += loss
           print('Epoch {0} Valid Loss : {1}'.format( epoch+1 ,vloss/numv) )
  def show(self):
    print('Accuracy is  :  {0}'.format(self.model.accuracy(self.xtrain , self.ytrain)))






if __name__ == '__main__':
  #inp = torch.randn(4)
  #test = normal(inputs = 4, node = 5 , out = 4)
  #print(inp.shape , test(inp))
  xt = torch.tensor([1,2,3,4,5,6,7,8,9] , dtype = torch.float)
  #xt = torch.randn(9)
  print(xt)
  print(xt.shape)
  yt = torch.tensor([0,0,1,0,0,1,0,0,1])
  
  test_model = normal(inputs = 1 , node = 10 , out = 2)
  test_model(xt)
  print(test_model.accuracy(xt , yt))
  w = test_model
  print(w)
  fullmodel = Setmodel(model = test_model  , xtrain  = xt , ytrain  = yt)
  #### Train ###
  print(fullmodel.show())
  fullmodel.train(epoch = 200 , bs = 3 , lr = 0.001)
  print(fullmodel.show())
