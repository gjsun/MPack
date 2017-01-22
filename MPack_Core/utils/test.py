import numpy as np

from .cosmology import *
from ..Ks_Mstar_Estimate import zmeans

def func1(x):
	return x * c_light
	
def func2(x):
	return x + zmeans