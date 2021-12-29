from numpy.lib.function_base import sinc
from model.simple import simple
import torch
import numpy as np
from experiment.exp import experiment
#import pandas as pd

if __name__=='__main__':

    exp=experiment()
    #exp.train()
    exp.test()
