Evolutionary Computation
Self Adaptation in Evolutionary Strategies

In order to run the experiment one must set the parameters for the script to run. To do this you should go to
the 'test.csv' file and change the values, if you wanna run several experiments at once, you just need to add a new line with
the parameters for each experiment.

The parameters in the file are explicit in the first line of the file, and are as such:
fitness_function,number_runs,number_of_generations,pop_size,prob_mutation,prob_crossover,tournament_size,type_cross,alpha,type_mutation,elite_percentage,test_id,type_algorithm,size_cromo


After settings the experiment parameters you run a python file. If you wish to run a Standard Evolutionary Algorithm run the 'standart.py' file, if you want the Self Adaptive one
run the 'adaptive.py' file.
