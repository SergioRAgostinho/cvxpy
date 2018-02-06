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

from cvxpy.problems.problem import Problem, Minimize
from cvxpy.expressions.constants import Parameter
from cvxpy.atoms import sum_squares

import numpy as np
from time import time
from collections import defaultdict
from multiprocessing import Process, Pipe

# Spectral step size.
def step_ls(p, d):
	sd = d.dot(d)/p.dot(d)   # Steepest descent
	mg = p.dot(d)/p.dot(p)   # Minimum gradient
	
	if 2*mg > sd:
		return mg
	else:
		return (sd - mg)

# Step size correlation.
def step_cor(p, d):
	return p.cor(d)/np.sqrt(p.dot(p)*d.dot(d))

# Safeguarding rule.
def step_safe(a, b, a_cor, b_cor, tau, eps = 0.2):
	if a_cor > eps and b_cor > eps:
		return np.sqrt(a*b)
	elif a_cor > eps and b_cor <= eps:
		return a
	elif a_cor <= eps and b_cor > eps:
		return b
	else:
		return tau

def step_spec(rho, du, dv, dl, dl_h, k, eps = 0.2, C = 1e10):
	# Compute spectral step sizes
	a_hat = step_ls(du, dl_h)
	b_hat = step_ls(dv, dl)
	
	# Estimate correlations
	a_cor = step_cor(du, dl_h)
	b_cor = step_cor(dv, dl)
	
	# Update step size
	scale = 1 + C/(1.0*k**2)
	rho_hat = step_safe(a_hat, b_hat, a_cor, b_cor, rho, eps)
	return max(min(rho_hat, scale*rho), rho/scale)

def proc_results(p_list, xbars):
	# TODO: Handle statuses.
	
	# Save primal values.
	pvars = [p.variables() for p in p_list]
	pvars = list(set().union(*pvars))
	for x in pvars:
		x.save_value(xbars[x.id])
	
	# TODO: Save dual values (for constraints too?).
	
	# Compute full objective.
	val = 0
	for p in p_list:
		# Flip sign of objective if maximization.
		if isinstance(p.objective, Minimize):
			val += p.objective.value
		else:
			val -= p.objective.value
	return val

def run_worker(p, rho, pipe):
	# Flip sign of objective if maximization.
	if isinstance(p.objective, Minimize):
		f = p.objective.args[0]
	else:
		f = -p.objective.args[0]
	cons = p.constraints
	
	# Add penalty for each variable.
	v = {}
	for xvar in p.variables():
		xid = xvar.id
		size = xvar.size
		v[xid] = {"x": xvar, "xbar": Parameter(size[0], size[1], value = np.zeros(size)),
				  "u": Parameter(size[0], size[1], value = np.zeros(size))}
		f += (rho/2.0)*sum_squares(xvar - v[xid]["xbar"] - v[xid]["u"]/rho)
	prox = Problem(Minimize(f), cons)
	
	# ADMM loop.
	while True:
		prox.solve()
		xvals = {}
		for xvar in prox.variables():
			xvals[xvar.id] = xvar.value
		pipe.send(xvals)
		
		# Update u += x - x_bar.
		xbars, i = pipe.recv()
		for key in v.keys():
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value += rho*(v[key]["x"].value - v[key]["xbar"].value)

def consensus(p_list, rho_list = None, max_iter = 100):
	# Number of problems.
	N = len(p_list)
	if rho_list is None:
		rho_list = N*[1.0]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (p_list[i], rho_list[i], remote))]
		procs[-1].start()

	# ADMM loop.
	start = time()
	for i in range(max_iter):
		# Gather and average x_i.
		xbars = defaultdict(float)
		xcnts = defaultdict(int)
		xvals = [pipe.recv() for pipe in pipes]
	
		for d in xvals:
			for key, value in d.items():
				xbars[key] += value
				++xcnts[key]
	
		for key in xbars.keys():
			if xcnts[key] != 0:
				xbars[key] /= xcnts[key]
	
		# Scatter x_bar.
		for pipe in pipes:
			pipe.send((xbars, i))
	end = time()

	[p.terminate() for p in procs]
	obj_val = proc_results(p_list, xbars)
	return obj_val, (end - start)

def solve_combined(p_list):
	obj = 0
	cons = []
	for p in p_list:
		if isinstance(p.objective, Minimize):
			f = p.objective.args[0]
		else:
			f = -p.objective.args[0]
		obj += f
		cons += p.constraints

	prob = Problem(Minimize(obj), cons)
	return prob.solve()

def basic_test():
	np.random.seed(1)
	m = 100
	n = 10
	max_iter = 10
	x = Variable(n)
	y = Variable(n/2)

	# Problem data.
	alpha = 0.5
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	b = A.dot(xtrue) + np.random.randn(m)

	# List of all the problems with objective f_i.
	p_list = [Problem(Minimize(sum_squares(A*x-b)), [norm(x,2) <= 1]),
			  Problem(Minimize((1-alpha)*sum_squares(y)/2))
			 ]
	N = len(p_list)   # Number of problems.
	pvars = [p.variables() for p in p_list]
	pvars = list(set().union(*pvars))   # Variables of problems.
	
	# Solve with consensus ADMM.
	obj_admm, elapsed = consensus(p_list, rho_list = N*[0.5], max_iter = max_iter)
	x_admm = [x.value for x in pvars]

	# Solve combined problem.
	obj_comb = solve_combined(p_list)
	x_comb = [x.value for x in pvars]

	# Compare results.
	for i in range(N):
		print "ADMM Solution:\n", x_admm[i]
		print "Base Solution:\n", x_comb[i]
		print "MSE: ", np.mean(np.square(x_admm[i] - x_comb[i])), "\n"
	print "ADMM Objective: %f" % obj_admm
	print "Base Objective: %f" % obj_comb
	print "Elapsed Time: %f" % elapsed

from cvxpy import *
basic_test()
