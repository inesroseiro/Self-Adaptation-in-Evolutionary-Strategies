__author__ = 'Ernesto Costa'
__date__ = 'March 2018'

import math
from utils import *
from sea_float_clean import *
import random

# Fitness
def merito_sphere(indiv):
    return sphere(fenotipo(indiv))
    
def merito_r(indiv):
    return rastrigin(fenotipo(indiv))
    
def merito_s(indiv):
    return schwefel(fenotipo(indiv))

def merito_q(indiv):
	return quartic(fenotipo(indiv))

def merito_step(indiv):
    return step(fenotipo(indiv))
    
def merito_rosenbrock(indiv):
	return rosenbrock(fenotipo(indiv))

def merito_griewangk(indiv):
	return griewangk(fenotipo(indiv))

def fenotipo(indiv):
    return indiv   

def sphere(indiv):
    """ De Jong F1 or the sphere function
	domain: [-5.12, 5.12] for each dimension.
	min = 0 at x = (0,0,...,0)
	"""
    return sum([ x ** 2 for x in indiv])


def rosenbrock(indiv):
	"""
	De Jong F2 or Rosembrock function
	domain: [-2.048, 2.048] for each dimension.
	min = 0 at x = (0,0,...,0)
	"""
	for i in range(len(indiv)-1):
		soma+= (1-indiv[i])**2 + 100*(indiv[i+1]-(individ[i])**2)**2
	return soma

def step(indiv):
	""" 
	De Jong F3 or the step function
	domain: [-5.12, 5.12] for each dimension.
	min = 0 at x = (0,0,...,0)
	"""
	soma=0
	tamanho = len(indiv)
	for i in range(len(indiv)):
		soma+= math.abs(indiv[i])

	return 6*tamanho +soma

def quartic(indiv):
    """
    quartic = DeJong 4
    domain = [-1.28; 1.28]
    minimum 0 at x = 0
    """
    y = sum([ (i+1) * x for i,x in enumerate(indiv)]) + random.uniform(0,1)
    return y

def rastrigin(indiv):
    """
    rastrigin function
    domain = [-5.12, 5.12]
    minimum at (0,....,0)
    """
    n = len(indiv)
    A = 10
    return A * n + sum([x**2 - A * math.cos(2 * math.pi * x) for x in indiv])

def schwefel(indiv):    
    """
    schwefel function
    domain = [-500; 500]
    minimum at (420.9687,...,420.9687)
    """
    y = sum([-x * math.sin(math.sqrt(math.fabs(x))) for x in indiv])
    return y


# based on https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/griewank.html
def griewangk(indiv):
	"""
	F6 Griewank's function
  	domain = [-600; 600]
    minimum at (0, ..., 0)
    """
	p1 = 0
	p2 = 1
	for i in range(len(indiv)):
		p1 += (indiv[i])**2
		p2 *= np.cos(indiv[i] / math.sqrt(i))
	return 1 + (p1+p2)/400





def get_params(func, size_cromo):
	if(func == 'quartic'):
		domain = [[-1.28,1.28]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/10
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_q
	elif(func=='rastrigin'):
		domain = [[-5.12,5.12]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/10
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_r
	elif(func=='schwefel'):
		domain = [[-500,500]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/100
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_s
	else:
		print('Error: unrecognized benchmark function <'+func+'>')
		exit()
    
    
if __name__ == '__main__':


	seeds = [2741, 8417, 5530, 4001, 1074, 828, 3878, 1652, 800, 1471, 3092, 2848, 6462, 7056, 7047, 4256, 4037, 6854, 918, 4042, 4333, 9051, 9126, 4210, 9385, 9860, 7732, 9063, 2044, 9998]
	size_cromo = 5
	path = '/Users/iroseiro/Desktop/CE_TP6/Self-Adaptation-in-Evolutionary-Strategies/plots/'
	plot_name = 'test2.png'
	#domain, sigma, fitness_func = get_params('quartic', size_cromo)
	#domain, sigma, fitness_func = get_params('rastrigin', size_cromo)
	domain, sigma, fitness_func = get_params('schwefel', size_cromo)

	boa,best_average, average_bests_gen = run(seeds,10,250,100,domain,0.01,sigma,0.9,tournament_selection(3),h_crossover(0.3,domain),muta_float_gaussian,sel_survivors_elite(0.1), fitness_func)

	display_stat_n(boa,best_average,path+plot_name)

	print(min(boa))
	

	    
	    

