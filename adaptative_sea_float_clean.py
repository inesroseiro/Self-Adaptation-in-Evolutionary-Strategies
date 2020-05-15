#! /usr/bin/env python

"""
Based on: sea_float.py
A very simple EA for float representation.
Ernesto Costa, February 2016

Adapted by:
Francisco Martins Ferreira
Maria Ines Roseiro
"""

import numpy as np
import matplotlib.pyplot as plt
from random import seed, random, randint, uniform, sample, shuffle, gauss
from operator import itemgetter


# ----------------------------------------- Initialize population -----------------------------------------
def gera_pop(size_pop, domain,sigma):
    x = [(gera_indiv_float_adaptative(domain,sigma), 0) for i in range(size_pop)]
    return x

def gera_indiv_float_adaptative(domain,sigma):
    indiv = [[uniform(domain[i][0], domain[i][1]) for i in range(len(domain))]]
    indiv.append(sigma)
    #print(indiv)
    return indiv
# -----------------------------------------  Variation operators ---------------------------------------------
# MUTATION - Gaussian float mutation
def muta_float_gene(gene, prob_muta, domain_i, sigma_i):
    value = random()
    new_gene = gene
    if value < prob_muta:
        # random val with gaussian distribution
        muta_value = gauss(0, sigma_i)
        new_gene = gene + muta_value
        # se estiver fora dos limites da função (fora do dominio) mantem-se o valor antigo, caso contrario, actualiza
        if new_gene < domain_i[0]:
            new_gene = domain_i[0]
        elif new_gene > domain_i[1]:
            new_gene = domain_i[1]
    return new_gene

def muta_float_gaussian(indiv, prob_muta, domain, sigma):
    cromo = indiv[:]

    for i in range(len(cromo[0])):
        cromo[0][i] = muta_float_gene(cromo[0][i], prob_muta, domain[i], cromo[1][i])
        cromo[1][i] = muta_float_sigma(prob_muta, cromo[1][i])
    return cromo

def muta_float_sigma(prob_muta,sigma_i):
    value = random()
    new_gene = sigma_i
    if value < prob_muta:
        # random val with gaussian distribution
        muta_value = gauss(0, sigma_i)
        new_gene = sigma_i + muta_value
        # se estiver fora dos limites da função (fora do dominio) mantem-se o valor antigo, caso contrario, actualiza
        if new_gene <= 0:
            new_gene = sigma_i
    return new_gene

# CROSSOVER
# Aritmetical  Crossover
def a_crossover(alpha):
    def aritmetical_crossover(indiv_1, indiv_2, prob_cross):
        size = len(indiv_1[0][0])
        value = random()
        if value < prob_cross:
            cromo_1 = indiv_1[0][0]
            cromo_2 = indiv_2[0][0]
            sigma_1 = indiv_1[0][1]
            sigma_2 = indiv_2[0][1]

            f1 = [[None] * size,[None] * size]
            f2 = [[None] * size,[None] * size]

            #print(cromo_1)

            for i in range(size):
                # a.c. formula = alpha * x1 + (1- alpha) * x2 (para o outro induvidual e ao contrario)
                f1[0][i] = alpha * cromo_1[i] + (1 - alpha) * cromo_2[i]
                f2[0][i] = (1 - alpha) * cromo_1[i] + alpha * cromo_2[i]

                f1[1][i] = alpha * sigma_1[i] + (1 - alpha) * sigma_2[i]
                f2[1][i] = (1 - alpha) * sigma_1[i] + alpha * sigma_2[i]
            return ((f1, 0), (f2, 0))
        return indiv_1, indiv_2

    return aritmetical_crossover


# based on: http://www.neurodimension.com/genetic/crossover.html
def h_crossover(alpha, domain):
    def heuristic_crossover(indiv_1, indiv_2, prob_cross):
        size = len(indiv_1[0])

        if random() < prob_cross:

            if indiv_2[1] < indiv_1[1]:
                best_cromo = indiv_1[0]
                worst_cromo = indiv_2[0]
            else:
                best_cromo = indiv_2[0]
                worst_cromo = indiv_1[0]

            f1 = [None] * size
            f2 = [None] * size
            for i in range(size):
                f1[i] = alpha * (worst_cromo[i] - best_cromo[i]) + best_cromo[i]
                f2[i] = best_cromo[i]

                f1[i] = constraint_domain(f1[i], domain[i])
                f2[i] = constraint_domain(f2[i], domain[i])

            return ((f1, 0), (f2, 0))
        return indiv_1, indiv_2

    return heuristic_crossover


# TOURNAMENT - Selection
def tournament_selection(tourn_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour(pop, tourn_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament


def tour(population, size):
    """Minimization Problem.Deterministic"""
    pool = sample(population, size)
    # print(pool)
    pool.sort(key=itemgetter(1))
    # print(pool)
    return pool[0]


# SURVIVALS - Elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring):
        # minimization
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism


# ----------------------------------------- Auxiliary -----------------------------------------
def best_pop(populacao):
    # minimization
    pop_size = len(populacao)
    populacao.sort(key=itemgetter(1))
    print(populacao[0])
    return populacao[0]


def average_pop(populacao):
    return sum([fit for cromo, fit in populacao]) / len(populacao)


# ----------------------------------------- Statistics -----------------------------------------


def run_adaptive(seeds, numb_runs, numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents, recombination,
        mutation, sel_survivors, fitness_func):
    statistics = []
    best_generations = []

    for i in range(numb_runs):
        seed(seeds[i])
        best, stat_best, stat_aver = sea_for_plot(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross,
                                                  sel_parents, recombination, mutation, sel_survivors, fitness_func)
        statistics.append(stat_best)
    stat_gener = list(zip(*statistics))
    boa = [min(g_i) for g_i in stat_gener]  # minimization
    aver_gener = [sum(g_i) / len(g_i) for g_i in stat_gener]
    for g_i in stat_gener:
        best_generations.append(min(g_i))

    average_best_gen = sum(best_generations) / len(best_generations)
    return boa, aver_gener, average_best_gen


def run_for_file(seeds, filename, numb_runs, numb_generations, size_pop, domain, prob_mut, sigma, prob_cross,
                 sel_parents, recombination, mutation, sel_survivors, fitness_func):
    with open(filename, 'w') as f_out:
        for i in range(numb_runs):
            seed(seeds[i])
            best = sea_float(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents,
                             recombination, mutation, sel_survivors, fitness_func)
            f_out.write(str(best[1]) + '\n')


# ----------------------------------------- Simple Ev Algorithms -----------------------------------------
# Simple [Float] Evolutionary Algorithm
def sea_float(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents, recombination, mutation,
              sel_survivors, fitness_func):
    populacao = gera_pop(size_pop, domain,sigma)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    for i in range(numb_generations):
        # parents selection
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, size_pop - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = recombination(indiv_1, indiv_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut, domain, sigma)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    return best_pop(populacao)


def sea_for_plot(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents, recombination, mutation,
                 sel_survivors, fitness_func):
    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, domain,sigma)
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
        for i in range(0, size_pop - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = recombination(indiv_1, indiv_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut, domain, sigma)
            #print("----------------------------")
            #print(novo_indiv)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    
        # Estatística
        stat.append(best_pop(populacao)[1])
        stat_aver.append(average_pop(populacao))

    return best_pop(populacao), stat, stat_aver
