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
import csv

# --------- Read from csv and assign to the variables -----------------
def read_test_file(test_file):    
    test_data = []
    with open(test_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                #print(len(row))
                correlation = row[0]
                max_value = int(row[1])
                amplitude = int(row[2])
                evaluation_func = row[3]
                numb_runs = int(row[4])
                number_of_generations = int(row[5])
                pop_size = int(row[6])
                number_of_items = int(row[7])
                prob_mutation = float(row[8])
                prob_crossover = float(row[9])
                tournament_size = int(row[10])
                type_crossover = row[11]
                type_mutation = row[12]
                elite_percentage = float(row[13])
                test_id = int(row[14])


                aux = [correlation,max_value,amplitude,evaluation_func,numb_runs,number_of_generations,pop_size,number_of_items,prob_mutation,prob_crossover,tournament_size,type_crossover,type_mutation,elite_percentage,test_id]

                #problem = generate_uncor(number_of_items,10)
                test_data.append(aux)

                line_count += 1
    return test_data

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

    seeds = [2741, 8417, 5530, 4001, 1074, 828, 3878, 1652, 800, 1471, 3092, 2848, 6462, 7056, 7047, 4256, 4037, 6854, 918, 4042, 4333, 9051, 9126, 4210, 9385, 9860, 7732, 9063, 2044, 9998]
    path = '/Users/iroseiro/Desktop/CE_TP6/Self-Adaptation-in-Evolutionary-Strategies/plots/'
    filename= 'test3.png'

    test_data = read_test_file('/Users/iroseiro/Desktop/CE_TP6/Self-Adaptation-in-Evolutionary-Strategies/tests/test.csv')
    #print(test_data)
    
    output_file = '/Users/iroseiro/Desktop/CE_TP6/Self-Adaptation-in-Evolutionary-Strategies/results2.txt'

    for i in range(len(test_data)):

        correlation = test_data[i][0]
        max_value = test_data[i][1]
        amplitude = test_data[i][2]
        evaluation_func = test_data[i][3]
        numb_runs = test_data[i][4]
        number_of_generations = test_data[i][5]
        pop_size = test_data[i][6]
        number_of_items = test_data[i][7]
        prob_mutation = test_data[i][8]
        prob_crossover = test_data[i][9]
        tournament_size = test_data[i][10]
        type_crossover = test_data[i][11]
        type_mutation = muta_bin
        elite_percentage = test_data[i][13]
        test_id = test_data[i][14]

        if(correlation == 'uncorrelated'):
            problem = generate_uncor(number_of_items,10)
        if(correlation == 'weak'):
                problem = generate_weak_cor(number_of_items,10, 5)
        if(correlation == 'strong'):
                problem = generate_strong_cor(number_of_items,10, 5)

        if(evaluation_func == 'evaluate_zero'):
            fit = merit(problem, evaluate_zero)
        if(evaluation_func == 'evaluate_log'):
            fit = merit(problem, evaluate_log)
        if(evaluation_func == 'evaluate_linear'):
            fit = merit(problem, evaluate_linear)
        if(evaluation_func == 'evaluate_quadratic'):
            fit = merit(problem, evaluate_quadratic)

        if(type_crossover == 'uniform_crossover'):
            type_crossover= uniform_cross
        if(type_crossover == 'one_point_cross'):
            type_crossover= one_point_cross
        if(type_crossover == 'two_points_cross'):
            type_crossover= two_points_cross

        if(type_crossover == 'muta_bin'):
            type_crossover= muta_bin
        if(type_crossover == 'swap_mutation'):
            type_crossover= swap_mutation

        print(test_id)
        #boa, best_average, average_bests_generation = run(seeds,numb_runs,number_of_generations,pop_size, number_of_items, prob_mutation, prob_crossover,tour_sel(tournament_size),type_crossover,type_mutation, sel_survivors_elite(elite_percentage),fit);
        run_for_file(seeds,output_file,numb_runs,number_of_generations,pop_size, number_of_items, prob_mutation, prob_crossover,tour_sel(tournament_size),type_crossover,type_mutation,sel_survivors_elite(elite_percentage),fit,str(test_id))
        
        #display_stat_n(boa, best_average, path+filename)
        #print(max(boa))

       #run(seeds,numb_runs,numb_generations,size_pop, domain, prob_mut, sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        if(type_cross == 'a_crossover'):  
            boa,best_average, average_bests_gen = run(seeds,number_runs,number_of_generations,pop_size,domain,prob_mutation,sigma,prob_crossover,tournament_selection(tournament_size),a_crossover(0.1),muta_float_gaussian,sel_survivors_elite(0.1), sphere)

        if(type_cross == 'h_crossover'):
            boa,best_average, average_bests_gen = run(seeds,number_runs,number_of_generations,pop_size,domain,prob_mutation,sigma,prob_crossover,tournament_selection(tournament_size),a_crossover(alpha),type_mutation,sel_survivors_elite(elite_percentage), fitness_function)
            #boa,best_average, average_bests_gen = run(seeds,number_runs,number_of_generations,pop_size,domain,prob_mutation,sigma,prob_crossover,tournament_selection(tournament_size),h_crossover(alpha,domain),type_mutation,sel_survivors_elite(elite_percentage), fitness_function)

    
    



