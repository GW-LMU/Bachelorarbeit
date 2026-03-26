from Traindata import train_data_problem, evaluate_candidate
from surrogate import train_surrogatmodell
from acquisition import train_acquisition
import torch
import cocoex
import random
import numpy as np


model = "GP"
model_acquisition = "ei"
n_iteration = 50
fun= 1                   
n_samples= 10             
value_range = (-5, 5)      
dimension = 2  



Train = train_data_problem(fun,n_samples,value_range,dimension)

train_X_tensor = Train[0]
train_Y_tensor = Train[1]

for i in range(n_iteration):
  trained_model = train_surrogatmodell(model, train_X_tensor,train_Y_tensor)

  trained_acquisition = train_acquisition(model_acquisition, trained_model, train_Y_tensor, value_range, dimension)                               # bzw. nöchster Punkt

  y_next = evaluate_candidate(trained_acquisition[0], fun, n_samples, value_range, dimension)

  train_X = torch.cat([train_X, trained_acquisition], dim=0)
  train_Y = torch.cat([train_Y, y_next], dim=0)