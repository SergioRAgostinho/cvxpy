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
from cvxpy import Variable, Problem, Minimize
from cvxpy.atoms import sum_squares, norm
from multiprocessing import Process, Pipe
from consensus_accel import *
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

def consensus_test():
	np.random.seed(1)
	m = 100
	n = 10
	MAX_ITER = 10
	x = Variable(n)
	y = Variable(n/2)

	# Problem data.
	alpha = 0.5
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	b = A.dot(xtrue) + np.random.randn(m)

	# List of all the problems with objective f_i.
	p_list = [Problem(Minimize(sum_squares(A*x-b)), [norm(x,2) <= 1]),
			  Problem(Minimize((1-alpha)*sum_squares(y)/2))]
	N = len(p_list)
	rho_init = N*[1.0]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = worker_map, args = (remote, p_list[i], rho_init[i]))]
		procs[-1].start()
	
	# Initial guesses.
	xbars = {x.id: np.zeros(x.size), y.id: np.zeros(y.size)}
	udicts = N*[{x.id: np.zeros(x.size), y.id: np.zeros(y.size)}]
	xuarr, xshapes = dicts_to_arr(xbars, udicts)
	
	# ADMM loop.
	for i in range(MAX_ITER):
		# xuarr = consensus_map(xuarr, pipes, xbars.keys(), xshapes, i)
		xuarr = anderson_accel(consensus_map, xuarr, m=5, g_args=(pipes, xbars.keys(), xshapes, i))
	
	[p.terminate() for p in procs]
	xbars, udicts = arr_to_dicts(xuarr, xbars.keys(), xshapes)
	for xid, xbar in xbars.items():
		print "Variable %d:\n" % xid, xbar

print "Basic Test:"
basic_test()

print "Consensus Test:"
consensus_test()
