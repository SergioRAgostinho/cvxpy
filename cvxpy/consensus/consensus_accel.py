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

from consensus import prox_step, x_average
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

def res_stop(res_ssq, eps = 1e-6):
	primal = np.sum([r["primal"] for r in res_ssq])
	dual = np.sum([r["dual"] for r in res_ssq])
	
	x_ssq = np.sum([r["x"] for r in res_ssq])
	xbar_ssq = np.sum([r["xbar"] for r in res_ssq])
	u_ssq = np.sum([r["u"] for r  in res_ssq])
	
	stopped = primal <= eps*max(x_ssq, xbar_ssq) and dual <= eps*u_ssq
	return primal, dual, stopped

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
		# Receive x_bar^(k) and u^(k).
		xbars, uvals, cur_iter = pipe.recv()
		
		ssq = {"primal": 0, "dual": 0, "x": 0, "xbar": 0, "u": 0}
		for key in v.keys():
			# Calculate primal/dual residual
			if v[key]["x"].value is None:
				primal = -xbars[key]
			else:
				primal = (v[key]["x"] - xbars[key]).value
			dual = (rho*(v[key]["xbar"] - xbars[key])).value
			
			# Set parameter values of x_bar^(k) and u^(k).
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value = uvals[key]
			
			# Save stopping rule criteria.
			ssq["primal"] += np.sum(np.square(primal))
			ssq["dual"] += np.sum(np.square(dual))
			if v[key]["x"].value is not None:
				ssq["x"] += np.sum(np.square(v[key]["x"].value))
			ssq["xbar"] += np.sum(np.square(v[key]["xbar"].value))
			ssq["u"] += np.sum(np.square(v[key]["u"].value))
		pipe.send(ssq)
		
		# Proximal step for x^(k+1) with x_bar^(k) and u^(k).
		prox.solve(*args, **kwargs)
		
		# Calcuate x_bar^(k+1).
		xvals = {k: np.asarray(d["x"].value) for k,d in v.items()}
		pipe.send((prox.status, xvals))
		xbars = pipe.recv()
		
		# Update u^(k+1) += rho*(x^(k+1) - x_bar^(k+1)).
		for key in v.keys():
			uvals[key] += rho.value*(v[key]["x"].value - xbars[key])
			
		# Return u^(k+1) and step size.
		pipe.send(uvals)

def consensus_map(xuarr, pipes, xids, xshapes, cur_iter):
	xbars, udicts = arr_to_dicts(xuarr, xids, xshapes)
	
	# Scatter x_bar^(k) and u^(k).
	N = len(pipes)
	for i in range(N):
		pipes[i].send((xbars, udicts[i], cur_iter))
	
	# Calculate normalized residuals.
	ssq = [pipe.recv() for pipe in pipes]
	primal, dual, stopped = res_stop(ssq)
	
	# Gather and average x^(k+1).
	prox_res = [pipe.recv() for pipe in pipes]
	xbars_n = x_average(prox_res)
	
	# Scatter x_bar^(k+1).
	for pipe in pipes:
		pipe.send(xbars_n)
	
	# Gather updated u^(k+1).
	udicts_n = [pipe.recv() for pipe in pipes]
	xuarr_n, xshapes = dicts_to_arr(xbars_n, udicts_n)
	return xuarr_n
