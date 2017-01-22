import numpy as np
import matplotlib.pyplot as plt
import hmf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import itertools

from utils.kvector_conversion import *

print k_CO_equiv_avg(k_CII=0.1,z_CII=6.,J=3)