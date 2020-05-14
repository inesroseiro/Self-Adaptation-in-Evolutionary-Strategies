#! /usr/bin/env python

"""
sea_float.py
A very simple EA for float representation.
Ernesto Costa, February 2016
"""

__author__ = 'Ernesto Costa'
__date__ = 'February 2016'

import numpy as np
import matplotlib.pyplot as plt
from random import seed,random,randint,uniform, sample, shuffle,gauss
from operator import itemgetter





# Auxiliary

# For the statistics
def run(seeds,numb_runs,numb_generations,size_pop, domain, prob_mut, sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    statistics = []
    for i in range(numb_runs):
        seed(seeds[i])
        best,stat_best,stat_aver = sea_for_plot(numb_generations,size_pop, domain, prob_mut, sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        statistics.append(stat_best)
    stat_gener = list(zip(*statistics))
    boa = [min(g_i) for g_i in stat_gener] # minimization
    aver_gener =  [sum(g_i)/len(g_i) for g_i in stat_gener]
    return boa,aver_gener
    
def run_for_file(seeds,filename,numb_runs,numb_generations,size_pop, domain,prob_mut, sigma,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    with open(filename,'w') as f_out:
        for i in range(numb_runs):
            seed(seeds[i])
            best= sea_float(numb_generations,size_pop, domain, prob_mut,sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
            f_out.write(str(best[1])+'\n')

# Simple [Float] Evolutionary Algorithm		
def sea_float(numb_generations,size_pop, domain, prob_mut, sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    """
    inicialize population: indiv = (cromo,fit)
    domain = [...-,[inf_i, sup_i],...]
    sigma = [..., sigma_i, ...]
    """
    
    populacao = gera_pop(size_pop,domain)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    for i in range(numb_generations):
        # parents selection
        mate_pool = sel_parents(populacao)
	# Variation
	# ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut, domain,sigma)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]     
    return best_pop(populacao)

def sea_for_plot(numb_generations,size_pop, domain, prob_mut,sigma,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,domain)
    # avalia população
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    
    # para a estatística
    stat = [best_pop(populacao)[1]]
    stat_aver = [average_pop(populacao)]
    
    for i in range(numb_generations):
        # selecciona progenitores
        mate_pool = sel_parents(populacao)
	# Variation
	# ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut,domain,sigma)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao] 
	
	# Estatística
        stat.append(best_pop(populacao)[1])
        stat_aver.append(average_pop(populacao))
	
    return best_pop(populacao),stat, stat_aver
