"""
Copyright 2018 Anqi Fu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from anderson import anderson_accel

def basic_test():
	def g(x):
		x = x[:,0]
		y0 = (2*x[0] + x[0]**2 - x[1])/2.0
		y1 = (2*x[0] - x[0]**2 + 8)/9 + (4*x[1]-x[1]**2)/4.0
		return np.array([[y0, y1]]).T

	m = 5
	x0 = np.array([[-1.0, 1.0]]).T
	res = anderson_accel(g, x0, m, max_iters=10, rcond=2)
	print(res)   # [-1.17397479  1.37821681]

basic_test()
