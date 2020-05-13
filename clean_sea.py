"""
Based on: sea_bin_visual.py
A very simple EA for binary representation.
Ernesto Costa, March 2015 & February 2016

Adapted by: 
Maria Inês Roseiro
Francisco Martins Ferreira
"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from random import seed,random,randint,uniform, sample, shuffle,gauss

# Initialize population
def gera_pop(size_pop,size_cromo):
    return [(gera_indiv(size_cromo),0) for i in range(size_pop)]

def gera_indiv(size_cromo):
    # random initialization
    indiv = [randint(0,1) for i in range(size_cromo)]
    return indiv

# ---------------------------------- Variation operators ----------------------------------

# Binary mutation	    
def muta_bin(indiv,prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i],prob_muta)
    return cromo

def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g

# Swap mutation	    
def swap_mutation(indiv, prob_mutation):
    cromo = indiv[:]
    for i in range(len(indiv)):

        value = random()
        if(value < prob_mutation):
            index = sample(range(len(cromo)),2)
            index1, index2 = index
            aux = cromo[index1]
            cromo[index1] = cromo[index2]
            cromo[index2] = cromo
            
        return cromo

# One point Crossover
def one_point_cross(indiv_1, indiv_2,prob_cross):
	value = random()
	if value < prob_cross:
	    cromo_1 = indiv_1[0]
	    cromo_2 = indiv_2[0]
	    pos = randint(0,len(cromo_1))
	    f1 = cromo_1[0:pos] + cromo_2[pos:]
	    f2 = cromo_2[0:pos] + cromo_1[pos:]
	    return ((f1,0),(f2,0))
	else:
	    return (indiv_1,indiv_2)

# Two points Crossover
def two_points_cross(indiv_1, indiv_2,prob_cross):
	value = random()
	if value < prob_cross:
	    cromo_1 = indiv_1[0]
	    cromo_2 = indiv_2[0]	    
	    pc= sample(range(len(cromo_1)),2)
	    pc.sort()
	    pc1,pc2 = pc
	    f1= cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
	    f2= cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
	    return ((f1,0),(f2,0))
	else:
	    return (indiv_1,indiv_2)


# Uniform Crossover 
def uniform_cross(indiv_1, indiv_2,prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1=[]
        f2=[]
        for i in range(0,len(cromo_1)):
            if random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])
        return ((f1,0),(f2,0))
    else:
        return (indiv_1,indiv_2)
	
# ----------------------------------- Parents Selection --------------------------------------------- 
# tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population,size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]


# -------------------------------------- Survivals Selection -----------------------------------------
# elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism



def best_pop(populacao):
    populacao.sort(key=itemgetter(1), reverse=True)
    return populacao[0]

def average_pop(populacao):
    return sum([fit for cromo,fit in populacao])/len(populacao)


def merito(indiv):
    # wrapper for fitness evaluation
    return evaluate(fenotipo(indiv))

def fenotipo(indiv):
    return indiv

def evaluate(indiv):
    return sum(indiv)

# ------------------------- SIMPLE EA -------------------------------------------------

# Simple [Binary] Evolutionary Algorithm		
def sea(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    for i in range(numb_generations):
        # sparents selection
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
        for indiv,fit in progenitores:
            novo_indiv = mutation(indiv,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]     
    return best_pop(populacao)


# Simple [Binary] Evolutionary Algorithm 
# Return the best plus, best by generation, average population by generation
def sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,size_cromo)
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
            cromo_1= mate_pool[i]
            cromo_2 = mate_pool[i+1]
            filhos = recombination(cromo_1,cromo_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for indiv,fit in progenitores:
            novo_indiv = mutation(indiv,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao] 
	
	# Estatística
        stat.append(best_pop(populacao)[1])
        stat_aver.append(average_pop(populacao))
    return best_pop(populacao), stat, stat_aver


# ------------------------- Run Function -------------------------------------------------

# For the statistics
def run(seeds,numb_runs,numb_generations,size_pop, size_cromo, prob_mut, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    statistics = []
    best_generations = []
    for i in range(numb_runs):
        seed(seeds[i])
        best,stat_best,stat_aver = sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        statistics.append(stat_best)
    stat_gener = list(zip(*statistics))
    boa = [max(g_i) for g_i in stat_gener] # maximization
    aver_gener =  [sum(g_i)/len(g_i) for g_i in stat_gener]
    #print(boa)
    for g_i in stat_gener:
        best_generations.append(max(g_i))

    average_best_gen = sum(best_generations)/len(best_generations)
    return boa,aver_gener, average_best_gen
    
def run_for_file (seeds,filename,numb_runs,numb_generations,size_pop, size_cromo, prob_mut, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    with open(filename,'w') as f_out:
        for i in range(numb_runs):
            seed(seeds[i])
            best = sea(numb_generations,size_pop,size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
            f_out.write(str(best[1])+'\n')


