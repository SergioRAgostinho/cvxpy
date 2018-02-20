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

import cvxpy.settings as s
from cvxpy.problems.problem import Problem, Minimize
from cvxpy.expressions.constants import Parameter
from cvxpy.atoms import sum_squares

from consensus import prox_step
from anderson import anderson_accel

import numpy as np
from collections import defaultdict

def dicts_to_arr(xbars, udicts):
	# Keep shape information.
	xshapes = [xbar.shape for xbar in xbars.values()]
	
	# Flatten x_bar and u into vectors.
	xflat = [xbar.flatten(order='C') for xbar in xbars.values()]
	uflat = [u.flatten(order='C') for udict in udicts for u in udict.values()]
	
	xuarr = np.concatenate(xflat + uflat)
	xuarr = np.array([xuarr]).T
	return xuarr, xshapes

def arr_to_dicts(arr, xids, xshapes):
	# Split array into x_bar and u vectors.
	xnum = len(xids)
	xelems = [np.prod(shape) for shape in xshapes]
	N = len(arr)/sum(xelems) - 1
	asubs = np.split(arr, N+1)
	
	# Reshape vectors into proper shape.
	sidx = 0
	xbars = []
	udicts = []
	for i in range(xnum):
		# Reshape x_bar.
		eidx = sidx + xelems[i]
		xvec = asubs[0][sidx:eidx]
		xbars += [np.reshape(xvec, xshapes[i])]
		
		# Reshape u_i for each pipe.
		uvals = []
		for j in range(N):
			uvec = asubs[j+1][sidx:eidx]
			uvals += [np.reshape(uvec, xshapes[i])]
		udicts += [uvals]
		sidx += xelems[i]
	
	# Compile into dicts.
	xbars = dict(zip(xids, xbars))
	udicts = [dict(zip(xids, u)) for u in udicts]
	return xbars, udicts
	
def worker_map(pipe, p, rho_init, *args, **kwargs):
	# Spectral step size parameters.
	spectral = kwargs.pop("spectral", False)
	Tf = kwargs.pop("Tf", 2)
	eps = kwargs.pop("eps", 0.2)
	C = kwargs.pop("C", 1e10)
	
	# Initiate proximal problem.
	prox, v, rho = prox_step(p, rho_init)
	
	# ADMM loop.
	while True:
		# Set parameter values.
		xbars, uvals, cur_iter = pipe.recv()
		for key in v.keys():
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value = uvals[key]
		
		# Proximal step with given x_bar and u.
		prox.solve(*args, **kwargs)
		
		# Update u += rho*(x - x_bar).
		for key in v.keys():
			uvals[key] += rho.value*(v[key]["x"].value - xbars[key])
			
		# Scatter x and updated u.
		xvals = {k: np.asarray(d["x"].value) for k,d in v.items()}
		pipe.send((prox.status, xvals, uvals))

def consensus_map(xuarr, pipes, xids, xshapes, cur_iter):
	xbars, udicts = arr_to_dicts(xuarr, xids, xshapes)
	
	# Scatter x_bar and u.
	N = len(pipes)
	for i in range(N):
		pipes[i].send((xbars, udicts[i], cur_iter))
	
	# Gather updated x and u.
	xbars_n = defaultdict(float)
	xcnts = defaultdict(int)
	udicts_n = []
	
	for i in range(N):
		status, xvals, uvals = pipes[i].recv()
		
		# Check if proximal step converged.
		if status in s.INF_OR_UNB:
			raise RuntimeError("Proximal problem is infeasible or unbounded")
		
		# Sum up x_i values.
		for key, value in xvals.items():
			xbars_n[key] += value
			++xcnts[key]
		udicts_n += [uvals]
	
	# Average x_i across pipes.
	for key in xbars.keys():
		if xcnts[key] != 0:
			xbars_n[key] /= xcnts[key]
	
	xuarr_n, xshapes = dicts_to_arr(dict(xbars_n), udicts_n)
	return xuarr_n
