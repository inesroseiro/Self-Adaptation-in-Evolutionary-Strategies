""" 
0/1 Knapsack Problem
Ernesto Costa, February 2018
"""

from clean_sea import *
import random
import copy
import operator
import math
from utils import *


# ---------  Evaluation -----------------
def merit(problem, eval_func):
    """
    problem is a dictionary describing the instance of the KP.
    keys = values (list of invidual's values), weigths (list of individual's weights),capacity (a float with the total capacity)
    """
    def fitness(indiv):
        quali = eval_func(phenotype(indiv))
        return quali
    return fitness


def phenotype(indiv):
    """from a binary string to a list of [id,weight,value]."""
    pheno = [[id, problem['weights'][id], problem['values'][id]] for id in range(len(indiv)) if indiv[id] == 1]
    return pheno

# ----------- Eval Functions ------------------

def evaluate_zero(feno):
    """ feno = [...,[ıd,weight,value],...]"""
    total_weight = sum([weight for id_,weight,value in feno])
    if total_weight > problem['capacity']:
        return 0
    return sum([value for id,weight,value in feno])

def evaluate_log(feno):
    """ feno = [...,[ıd,weight,value],...]"""
    total_weight = sum([weight for id_,weight,value in feno])
    quality = sum([value for id,weight,value in feno])
    capacity = problem['capacity']
    if total_weight > capacity:
        rho = max([v/w for i,w,v in feno])
        quality -= math.log(1 + rho * (total_weight - capacity),2)
    return quality

def evaluate_linear(feno):
    """ feno = [...,[ıd,weight,value],...]"""
    total_weight = sum([weight for id_,weight,value in feno])
    quality = sum([value for id,weight,value in feno])
    capacity = problem['capacity']
    if total_weight > capacity:
        rho = max([v/w for i,w,v in feno])
        quality -=  rho * (total_weight - capacity)
    return quality

def evaluate_quadratic(feno):
    """ feno = [...,[ıd,weight,value],...]"""
    total_weight = sum([weight for id_,weight,value in feno])
    quality = sum([value for id,weight,value in feno])
    capacity = problem['capacity']
    if total_weight > capacity:
        rho = max([v/w for i,w,v in feno])
        quality -=  (rho * (total_weight - capacity))**2
    return quality

# ------------ Data Sets -----------------------
def generate_uncor(size_items,max_value):
    weights = [random.uniform(1,max_value) for i in range(size_items)]
    values = [random.uniform(1,max_value) for i in range(size_items)]
    capacity = int(0.5 * sum(weights))
    return {'weights':weights, 'values':values, 'capacity': capacity}

def generate_weak_cor(size_items,max_value, amplitude):
    weights = [random.uniform(1,max_value) for i in range(size_items)]
    values = []
    for i in range(size_items):
        value = weights[i] + random.uniform(-amplitude,amplitude)
        while value <= 0:
            value = weights[i] + random.uniform(-amplitude,amplitude)
        values.append(value)
    capacity = int(0.5 * sum(weights))
    return {'weights':weights, 'values':values, 'capacity': capacity}

def generate_strong_cor(size_items,max_value,amplitude):
    weights = [random.uniform(1,max_value) for i in range(size_items)]
    values = [weights[i] + amplitude for i in range(size_items)]
    capacity = int(0.5 * sum(weights))
    return {'weights':weights, 'values':values, 'capacity': capacity}

  
# ------------- Repair individuals -------------------------
def repair_weight(cromo,problem):
    """repair an individual be eliminating items using the least weighted gene."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv)
    pheno.sort(key= operator.itemgetter(1))
    
    weight_indiv = get_weight(indiv,problem)
    for index, weight,value in pheno:
        if weight_indiv <= capacity:
            break
        else:
            indiv[index] = 0
            weight_indiv -= weight
    return indiv
  
def repair_value(cromo,problem):
    """repair an individual be eliminating items using the least valued gene."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv)
    pheno.sort(key= operator.itemgetter(2))
    
    weight_indiv = get_weight(indiv,problem)
    for index, weight,value in pheno:
        if weight_indiv <= capacity:
            break
        else:
            indiv[index] = 0
            weight_indiv -= weight  
    return indiv

def repair_value_to_profit(cromo,problem):
    """repair an individual be eliminating items using the ratio value/weight."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv)
    pheno = [[i,w,v, float(v/w)] for i,w,v in pheno] 
    pheno.sort(key= operator.itemgetter(3))
    
    weight_indiv = get_weight(indiv,problem)
    for index, weight,value,ratio in pheno:
        if weight_indiv <= capacity:
            break
        else:
            indiv[index] = 0
            weight_indiv -= weight  
    return indiv
            
# -------------- Auxiliary ------------------------            
def get_weight(indiv,problem):
    total_weight = sum([ problem['weights'][gene] for gene in range(len(indiv)) if indiv[gene] == 1])
    return total_weight

def get_value(indiv,problem):
    total_value = sum([ problem['values'][gene] for gene in range(len(indiv)) if indiv[gene] == 1])
    return total_value	 


if __name__ == '__main__':
    number_of_generations = 10
    number_of_items = 20
    pop_size = 10
    prob_mutation = 0.05
    prob_crossover = 0.8
    tournament_size = 3
    seeds = [2741, 8417, 5530, 4001, 1074, 828, 3878, 1652, 800, 1471, 3092, 2848, 6462, 7056, 7047, 4256, 4037, 6854, 918, 4042, 4333, 9051, 9126, 4210, 9385, 9860, 7732, 9063, 2044, 9998]
    numb_runs = 10
    elite_percentage = 0.02

    problem = generate_uncor(number_of_items,10)
    #problem = generate_weak_cor(number_of_items,10, 5)
    #problem = generate_strong_cor(number_of_items,10, 5)

    path = '/Users/iroseiro/Desktop/CE_TP6/Self-Adaptation-in-Evolutionary-Strategies/plots/'
    filename= 'test3.png'
    
    # DIFFERENT FITNESS FUNCTIONS
    fit = merit(problem, evaluate_zero)
    
    #best_ind, best_1, average_1 = sea_for_plot(number_of_generations, pop_size, number_of_items, prob_mutation, prob_crossover,tour_sel(tournament_size),one_point_cross,muta_bin, sel_survivors_elite(elite_percentage), fit)
    
    boa, best_average, average_bests_generation = run(seeds,numb_runs,number_of_generations,pop_size, number_of_items, prob_mutation, prob_crossover,tour_sel(tournament_size),one_point_cross,muta_bin, sel_survivors_elite(elite_percentage),fit);

    #display(best_ind, phenotype)
    display_stat_n(boa, best_average, path+filename)

    print(max(boa))
    
    #run(seeds,numb_runs,numb_generations,size_pop, size_cromo, prob_mut, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
    



