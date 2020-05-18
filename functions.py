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
	soma = 0
	for i in range(len(indiv)-1):
		soma+= (1-indiv[i])**2 + 100*(indiv[i+1]-(indiv[i])**2)**2
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
		soma+= math.fabs(indiv[i])

	return 6*tamanho +soma

def quartic(indiv):
    """
    quartic = DeJong 4
    domain = [-1.28; 1.28]
    minimum 0 at x = 0
    """
    y = sum([ (i+1) * x**4 for i,x in enumerate(indiv)]) + random.uniform(0,1)
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
		p2 *= np.cos(indiv[i] / math.sqrt(i+1))
	return 1 + p1/4000 + p2


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
	elif(func=='sphere'):
		domain = [[-5.12,5.12]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/10
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_sphere
	elif(func=='rosenbrock'):
		domain = [[-5.12,5.12]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/10
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_rosenbrock
	elif(func=='step'):
		domain = [[-5.12,5.12]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/10
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_step
	elif(func=='griewangk'):
		domain = [[-600,600]]*size_cromo
		amp_sigma = (domain[0][1]-domain[0][0])/100
		sigma = [round(random.uniform(0,amp_sigma),2)  for i in range(size_cromo)]
		return domain, sigma, merito_griewangk
	else:
		print('Error: unrecognized benchmark function <'+func+'>')
		exit()
    
    
if __name__ == '__main__':
	pass
	

	    
	    

