#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 07:30:07 2022
@author: ritwiksadhu, sbnietert
"""

## Experiments on sliced wasserstein distances

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from multiprocessing import Pool

# config
folder = '/home/ubuntu/data/sliced-wp-experiments'
do_exp1 = do_exp2 = do_exp3 = True

## Global experiment settings
n_samples = (10,20,50,100,200,500,1000)
simul = 100
bootstrap_re = 10
dims = (2,5,10,20,50,100)
n_steps = (10,20,50,100,200,500,1000)
important_n_samples = [1000]
important_n_steps = [1000]

## Below code will save to disk output of each experiment as numpy objects.

# Experiment 1: Normal vs mixture of normals with same mean
if do_exp1:
    def simulate_sw2sq_exp1(N, d, m):
        X = np.random.rand(N,d)
        delta = np.random.binomial(1, 0.5, size = N)
        Y = np.random.multivariate_normal(mean = np.zeros(d), 
                                        cov = np.diag(np.ones(d)) + 
                                        0.5*np.matmul(np.ones((d,1)), np.ones((1,d)))/d,
                                        size = N)
        Z = np.random.randn(N,d)
        Y = [np.array([Y,Z])[delta[i],i,:] for i in range(N)]
        return(sw_distance(X, Y, m)**2)

    pop_N = 5000
    pop_m = 2000 # approximate population value by taking N and m large

    print('estimating population values')
    sw2sq_exp1_pop = [0] * len(dims)
    for i,d in enumerate(dims):
        print(f'd: {d}')
        sw2sq_exp1_pop[i] = simulate_sw2sq_exp1(pop_N, d, pop_m)

    print('estimating errors')
    errors = np.empty((len(n_steps), len(dims), len(n_samples), simul))
    errors[:] = np.nan
    params = []
    for i,m in enumerate(n_steps):
        for j,d in enumerate(dims):
            for k,N in enumerate(n_samples):
                if m in important_n_steps or N in important_n_samples:
                    params.append((i,m,j,d,k,N))

    def comp_errors(param):
        i,m,j,d,k,N = param
        print(m,d,N)
        output = np.zeros(simul)
        for l in range(simul):
            output[l] = abs(simulate_sw2sq_exp1(N,d,m) - sw2sq_exp1_pop[j])
        return (param, output)

    print('starting parallel processing')
    with Pool(6) as p:
        outputs = p.map(comp_errors, params)

    print('reformatting to error array')
    for param,output in outputs:
        i,m,j,d,k,N = param
        for l in range(simul):
            errors[i,j,k,l] = output[l]
                        
    np.save(timestamped_path(folder,'exp1_errors','npy'),errors)
    del errors


# Experiment 2: Normal with mean difference
if do_exp2:
    def simulate_sw2sq_error_exp2(N, d, m):
        X = np.random.randn(N, d)+2
        Y = np.random.randn(N,d)
        sw_dist_emp = sw_distance(X, Y, m)**2
        return abs(sw_dist_emp - 4)

    print('estimating errors')
    errors = np.empty((len(n_steps), len(dims), len(n_samples), simul))
    errors[:] = np.nan
    params = []
    for i,m in enumerate(n_steps):
        for j,d in enumerate(dims):
            for k,N in enumerate(n_samples):
                if m in important_n_steps or N in important_n_samples:
                    params.append((i,m,j,d,k,N))

    def comp_errors(param):
        i,m,j,d,k,N = param
        print(m,d,N)
        output = np.zeros(simul)
        for l in range(simul):
            output[l] = simulate_sw2sq_error_exp2(N, d, m)
        return (param, output)

    print('starting parallel processing')
    with Pool(6) as p:
        outputs = p.map(comp_errors, params)

    print('reformatting to error array')
    for param,output in outputs:
        i,m,j,d,k,N = param
        for l in range(simul):
            errors[i,j,k,l] = output[l]
                        
    np.save(timestamped_path(folder,'exp2_errors','npy'),errors)
    del errors


# Experiment 3: Two ten component normal mixture models
if do_exp3:
    from scipy.stats import multivariate_normal
    def cov_gen(x):
        if(np.ndim(x) == 1):
            x= np.vstack(x).T
        return np.matmul(x.T, x)

    mean1_list = [multivariate_normal(mean = np.ones(d)).rvs(size=10) for d in dims]
    mean2_list = [multivariate_normal(mean = 3*np.ones(d)).rvs(size=10) for d in dims]

    ranks = [np.random.randint(low = 1, high = d+1, size=10) for d in dims]
    cov1_list = [[cov_gen(multivariate_normal(mean = np.zeros(dims[i])).rvs(size = ranks[i][j])) / (ranks[i][j]) 
                for j in range(10)] for i in range(len(dims))]
    cov2_list = [[cov_gen(multivariate_normal(mean = np.zeros(dims[i])).rvs(size = ranks[i][j])) / (ranks[i][j])
                for j in range(10)] for i in range(len(dims))]

    def gmm_sample(means, covs, size=1):
        out_array = np.zeros((size, len(means[0])))
        
        for i in range(size):
            comp = np.random.randint(10)
            out_array[i] = multivariate_normal(means[comp], covs[comp], allow_singular = True).rvs()
        return(out_array)

    def simulate_sw2sq_exp3(N, dim_i, dims, mean1_list, cov1_list, mean2_list, cov2_list, m):
        d = dims[dim_i]
        # print(f"N={N}, d = {d}, m={m}")
        X = gmm_sample(mean1_list[dim_i], cov1_list[dim_i], size = N)
        Y = gmm_sample(mean2_list[dim_i], cov2_list[dim_i], size = N)
        return(sw_distance(X, Y, m)**2)

    pop_N = 5000
    pop_m = 2000 

    print('estimating population values')
    sw2sq_exp3_pop = [0] * len(dims)
    for i,d in enumerate(dims):
        print(f'd: {d}')
        sw2sq_exp3_pop[i] = simulate_sw2sq_exp3(pop_N, i, dims, mean1_list, cov1_list, mean2_list, cov2_list, pop_m)
    
    print('estimating errors')
    errors = np.empty((len(n_steps), len(dims), len(n_samples), simul))
    errors[:] = np.nan
    params = []
    for i,m in enumerate(n_steps):
        for j,d in enumerate(dims):
            for k,N in enumerate(n_samples):
                if m in important_n_steps or N in important_n_samples:
                    params.append((i,m,j,d,k,N))

    def comp_errors(param):
        i,m,j,d,k,N = param
        print(m,d,N)
        output = np.zeros(simul)
        for l in range(simul):
            output[l] = abs(simulate_sw2sq_exp3(N, j, dims, mean1_list, cov1_list, mean2_list, cov2_list, m) - sw2sq_exp3_pop[j])
        return (param, output)

    print('starting parallel processing')
    with Pool(6) as p:
        outputs = p.map(comp_errors, params)

    print('reformatting to error array')
    for param,output in outputs:
        i,m,j,d,k,N = param
        for l in range(simul):
            errors[i,j,k,l] = output[l]
                        
    np.save(timestamped_path(folder,'exp3_errors','npy'),errors)
    del errors