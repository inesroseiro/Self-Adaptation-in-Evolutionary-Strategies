#! /usr/bin/env python

"""
Work by:
Francisco Martins Ferreira
Maria Ines Roseiro
"""

import math
from utils import *
from sea_float_clean import *
import random
from functions_adaptative import *
import csv
from adaptative_sea_float_clean import *

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
                fitness_function= row[0]
                number_runs= int(row[1])
                number_of_generations= int(row[2])
                pop_size= int(row[3])
                prob_mutation = float(row[4])
                prob_crossover = float(row[5])
                tournament_size = int(row[6])
                type_cross = row[7]
                alpha = float(row[8])
                type_mutation = row[9]
                elite_percentage = float(row[10])
                test_id = row[11]
                type_algorithm = row[12]
                size_cromo = int(row[13])
    
                aux = [fitness_function,number_runs,number_of_generations,pop_size,prob_mutation,prob_crossover,tournament_size,type_cross,alpha,type_mutation,elite_percentage,test_id,type_algorithm,size_cromo]

                test_data.append(aux)
                line_count += 1
    return test_data


if __name__ == '__main__':
    seeds = [2741, 8417, 5530, 4001, 1074, 828, 3878, 1652, 800, 1471, 3092, 2848, 6462, 7056, 7047, 4256, 4037, 6854, 918, 4042, 4333, 9051, 9126, 4210, 9385, 9860, 7732, 9063, 2044, 9998]

    path = 'plots/'
    plot_name = 'test2.png'
    test_data = read_test_file('test.csv')
    filename ='standart_results.txt'
    #print(test_data)
    for i in range(len(test_data)):
        fitness_function= test_data[i][0]
        print(fitness_function)
        number_runs= test_data[i][1]
        number_of_generations= test_data[i][2]
        pop_size= test_data[i][3]
        prob_mutation = test_data[i][4]
        prob_crossover = test_data[i][5]
        tournament_size = test_data[i][6]
        type_cross = test_data[i][7]
        alpha = test_data[i][8]
        type_mutation = test_data[i][9]
        elite_percentage = test_data[i][10]
        test_id = test_data[i][11]
        type_algorithm = test_data[i][12]
        size_cromo = test_data[i][13]

        domain, sigma, fitness_func = get_params(fitness_function, size_cromo)


        #run(seeds,numb_runs,numb_generations,size_pop, domain, prob_mut, sigma, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        if(type_cross == 'a_crossover'):  
            #boa,best_average, average_bests_gen = run_adaptive(seeds,number_runs,number_of_generations,pop_size,domain,prob_mutation,sigma,prob_crossover,tournament_selection(tournament_size),a_crossover(alpha),muta_float_gaussian,sel_survivors_elite(elite_percentage), fitness_func)
            run_for_file(seeds,filename,number_runs,number_of_generations,pop_size,domain,prob_mutation, sigma,prob_crossover,tournament_selection(tournament_size),a_crossover(alpha),muta_float_gaussian,sel_survivors_elite(elite_percentage),fitness_func,test_id)

        if(type_cross == 'h_crossover'):
            run_for_file(seeds,filename,number_runs,number_of_generations,pop_size,domain,prob_mutation, sigma,prob_crossover,tournament_selection(tournament_size),h_crossover(alpha,domain),muta_float_gaussian,sel_survivors_elite(elite_percentage),fitness_func,test_id)
            #boa,best_average, average_bests_gen = run(seeds,number_runs,number_of_generations,pop_size,domain,prob_mutation,sigma,prob_crossover,tournament_selection(tournament_size),h_crossover(alpha,domain),muta_float_gaussian,sel_survivors_elite(elite_percentage), fitness_func)

            

        #display_stat_n(boa,best_average,path+plot_name)
        #print(min(boa))
