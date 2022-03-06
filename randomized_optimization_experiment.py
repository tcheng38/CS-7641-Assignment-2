# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:57:59 2022

@author: cheng164
"""

import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlrose_hiive
import random
import numpy as np
import pandas as pd
import time
import itertools
from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx

np.random.seed(1)
random.seed(1)


#%%
def para_tuning(algorithm_name, problem, fixed_para, tunable_para, maximize):
    
    hyper_tuning_history = {}
    best_param = None
    best_fitness_value = None
    tunable_para_names = list(tunable_para.keys())
    tunable_para_values = list(tunable_para.values())
    
    
    for i in itertools.product(*tunable_para_values):
        
        if algorithm_name == 'RHC':
            best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                                                 max_attempts = fixed_para['max_attempts'], 
                                                                                 max_iters= fixed_para['max_iters'], 
                                                                                 init_state= fixed_para['initial_state'],
                                                                                 curve=True,
                                                                                 random_state = 1,
                                                                                 restarts = i[0])
        elif algorithm_name == 'SA':
            schedule= mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=i[0], min_temp=i[1])
            best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem, 
                                                                                max_attempts = fixed_para['max_attempts'], 
                                                                                max_iters= fixed_para['max_iters'],
                                                                                init_state= fixed_para ['initial_state'],
                                                                                curve=True, 
                                                                                random_state = 1,
                                                                                schedule=schedule)
            
        elif algorithm_name == 'GA':
           best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem, 
                                                                              max_attempts = fixed_para['max_attempts'], 
                                                                              max_iters= fixed_para['max_iters'],
                                                                              pop_size=i[0],
                                                                              random_state = 1,
                                                                              mutation_prob=i[1])
            
            
        elif algorithm_name == 'MIMIC':
#            print ('i=', i)
            try:
                best_state, best_fitness, fitness_curve = mlrose_hiive.mimic(problem, 
                                                                         max_attempts = fixed_para['max_attempts'], 
                                                                         max_iters= fixed_para['max_iters'],
                                                                         pop_size=i[0],
                                                                         keep_pct=i[1],
                                                                         random_state = 1
                                                                         )
            
            
            except ValueError:
                print("Oops!  That was math domain error: (pop_size, keep_pct)=", i)
                best_state, best_fitness, fitness_curve = None, None, None
                pass
    
        if tuple(tunable_para_names) not in hyper_tuning_history:
            hyper_tuning_history[tuple(tunable_para_names)]=[(i, best_fitness)]   # Store the hyperparameter tuning history to a dic
        else:
            hyper_tuning_history[tuple(tunable_para_names)].append((i, best_fitness))
                                                                                                 
        if not best_fitness_value:
            best_param = i
            best_fitness_value = best_fitness
        elif maximize is True and best_fitness > best_fitness_value:
            best_param = i
            best_fitness_value = best_fitness
        elif maximize is not True and best_fitness < best_fitness_value:   # In case need to minimize the fitness function
            best_param = i
            best_fitness_value = best_fitness 
    
    best_param_dic = dict(zip(tunable_para_names, best_param))
    print("Best parameters for {} is: {} . Best fitness value = {}".format(algorithm_name, best_param_dic, best_fitness_value))    
    
    return best_param_dic, best_fitness_value, hyper_tuning_history
    

    
def iter_results_plot(str_name, max_iter, fitness_curves, running_times, input_size):
        # Plot Fitness vs Iterations
    iterations = range(1, max_iter+1)
    algor_name = ['RHC', 'SA', 'GA', 'MIMIC']
    color = ['green', 'red', 'blue', 'black', 'orange']
    
    for i in range(0, len(fitness_curves)):
        curve_padded = np.pad(fitness_curves[i][:,0], (0, max_iter -len(fitness_curves[i])), 'edge')   # pad the curve using edge value
        plt.plot(iterations, curve_padded, label=algor_name[i], color=color[i])
    
    plt.figure()
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title('Fitness Curve for '+ str_name + " @ Input Size =" + str(input_size))
    plt.savefig("results/"+ str_name + "_fitness_curve.png")

    # Plot Time Table
    data = [('RHC', round(running_times[0], 5)), 
            ('SA', round(running_times[1], 5)), 
            ('GA', round(running_times[2], 5)), 
            ('MIMIC', round(running_times[3], 5))] 
    
    df = pd.DataFrame(data, columns =['Algorithm', 'Time (s)']) 
    dfi.export(df,"results/"+ str_name +"_running_time.png")



def input_size_plot(str_name, input_size_list, fitness_curves_list, running_time_list):
        
    # Plot fitness score  vs input size
    plt.figure() 
    plt.plot(np.array(input_size_list), np.array(fitness_curves_list)[:,0], marker="o", label='RHC', color='green')
    plt.plot(np.array(input_size_list), np.array(fitness_curves_list)[:,1], marker="o", label='SA', color='red')
    plt.plot(np.array(input_size_list), np.array(fitness_curves_list)[:,2], marker="o", label='GA', color='blue')
    plt.plot(np.array(input_size_list), np.array(fitness_curves_list)[:,3], marker="o", label='MIMIC', color='black')
    plt.legend(loc="best")
    plt.xlabel("Input Size")
    plt.ylabel("Best Fitness Score")
    plt.title('Best Fitness Scores vs.Input Size for '+ str_name)
    plt.savefig("results/"+ str_name + "_best_fitness_Score.png")

    # Plot Running time  vs input size
    plt.figure() 
    plt.yscale("log") 
    plt.plot(np.array(input_size_list), np.array(running_time_list)[:,0], marker="o", label='RHC', color='green')
    plt.plot(np.array(input_size_list), np.array(running_time_list)[:,1], marker="o", label='SA', color='red')
    plt.plot(np.array(input_size_list), np.array(running_time_list)[:,2], marker="o", label='GA', color='blue')
    plt.plot(np.array(input_size_list), np.array(running_time_list)[:,3], marker="o", label='MIMIC', color='black')
    plt.legend(loc="best")
    plt.xlabel("Input Size")
    plt.ylabel("Running Time (sec)")
    plt.title('Running Times vs.Input Size for '+ str_name)
    plt.savefig("results/"+ str_name + "_running_time.png")
    
    

#%%
def run_four_peaks(Tuning_flag, bit_length, max_attempts, max_iters):
    print("Running Four Peaks Randomized Optimization Experiment: \n")

    hyper_tuning_history_list = []

    # Define Fitness function and problem object
    ini_state = np.random.randint(2,size=bit_length)
    
    fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
    problem = mlrose_hiive.DiscreteOpt(length=bit_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [50, 80, 100, 150]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, hyper_tuning_history_RHC = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_RHC)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.0001, 0.001, 0.01, 0.05, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, hyper_tuning_history_SA = para_tuning('SA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_SA)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400, 600], 'mutation_prob': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, hyper_tuning_history_GA = para_tuning('GA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_GA)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1
                                                        )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'keep_pct': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, hyper_tuning_history_MIMIC = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=True) 
        hyper_tuning_history_list.append(hyper_tuning_history_MIMIC)
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1
                                                                )
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if bit_length == 100:
        iter_results_plot('fourpeaks', max_iters, fitness_curves, running_time, input_size = bit_length)

    return fitness_curves, running_time, fitness_scores, hyper_tuning_history_list 

#%%
def run_flipflop(Tuning_flag, bit_length, max_attempts, max_iters):
    
    print("Running Flip Flop Randomized Optimization Experiment: \n")

    hyper_tuning_history_list = []
    # Define Fitness function and problem object
    ini_state = np.random.randint(2,size=bit_length)
    
    fitness = mlrose_hiive.FlipFlop()
    problem = mlrose_hiive.DiscreteOpt(length=bit_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [50, 80, 100, 150]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, hyper_tuning_history_RHC = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_RHC)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.0001, 0.001, 0.01, 0.05, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, hyper_tuning_history_SA = para_tuning('SA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_SA)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'mutation_prob': [0.2, 0.4, 0.6]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, hyper_tuning_history_GA = para_tuning('GA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_GA)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1,
                                                        )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'keep_pct': [0.2, 0.4, 0.6]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, hyper_tuning_history_MIMIC = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=True) 
        hyper_tuning_history_list.append(hyper_tuning_history_MIMIC)
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1
                                                                )
    
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if bit_length == 100:
        iter_results_plot('FlipFlop', max_iters, fitness_curves, running_time, input_size = bit_length)

    return fitness_curves, running_time, fitness_scores, hyper_tuning_history_list

#%%
def run_one_max(Tuning_flag, bit_length, max_attempts, max_iters):
    
    print("Running One Max Randomized Optimization Experiment: \n")

    # Define Fitness function and problem object
    ini_state = np.random.randint(2,size=bit_length)
    
    fitness = mlrose_hiive.OneMax()
    problem = mlrose_hiive.DiscreteOpt(length=bit_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [20, 40, 60, 80, 100]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, _ = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=True)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 20
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.001, 0.005, 0.01, 0.05, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, _ = para_tuning('SA', problem, fixed_para, tunable_para, maximize=True)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [100, 200, 400], 'mutation_prob': [0.2, 0.4, 0.8]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, _ = para_tuning('GA', problem, fixed_para, tunable_para, maximize=True)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 100, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'keep_pct': [0.2, 0.4, 0.6]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, _ = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=True) 
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1)
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if bit_length == 100:
        iter_results_plot('OneMax', max_iters, fitness_curves, running_time, input_size = bit_length)

    return fitness_curves, running_time, fitness_scores 

#%%
def Run_N_Queen(Tuning_flag, num_queens, max_attempts, max_iters):
    
    print("Running N_Queen Randomized Optimization Experiment: \n")

    # Define Fitness function and problem object
    ini_state = np.array(random.sample(range(num_queens), num_queens))
    
    fitness = mlrose_hiive.Queens()
    problem = mlrose_hiive.DiscreteOpt(length=num_queens, fitness_fn=fitness, maximize=False, max_val=num_queens)
    problem.set_mimic_fast_mode(True)

     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [50, 80, 100, 150]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, _ = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=False)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.0001, 0.001, 0.01, 0.05, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, _ = para_tuning('SA', problem, fixed_para, tunable_para, maximize=False)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [100, 200, 400], 'mutation_prob': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, _ = para_tuning('GA', problem, fixed_para, tunable_para, maximize=False)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400, 600], 'keep_pct': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is False:
        best_param_MIMIC, best_fitness_value_MIMIC, _ = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=False) 
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1,
                                                                )
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if num_queens ==100: 
        iter_results_plot('N_Queen', max_iters, fitness_curves, running_time, input_size = num_queens)

    return fitness_curves, running_time, fitness_scores 



#%%

def Run_TSP(Tuning_flag, n_locations, max_attempts, max_iters):
    
    print("Running TSP Randomized Optimization Experiment: \n")

    hyper_tuning_history_list = []
    # Define Fitness function and problem object
    coord_lists = np.random.randint(0, 50, size=(n_locations, 2)).tolist()
    ini_state = np.array(random.sample(range(n_locations), n_locations))

    fitness = mlrose_hiive.TravellingSales(coords = coord_lists)
    problem = mlrose_hiive.TSPOpt(length=n_locations, fitness_fn=fitness, maximize=False)
    problem.set_mimic_fast_mode(True)

     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [20, 40, 60, 80, 100]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, hyper_tuning_history_RHC = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_RHC)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.001, 0.005, 0.01, 0.05, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, hyper_tuning_history_SA = para_tuning('SA', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_SA)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'mutation_prob': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, hyper_tuning_history_GA = para_tuning('GA', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_GA)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        #random_state = 1,
                                                        )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400], 'keep_pct': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, hyper_tuning_history_MIMIC = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=False) 
        hyper_tuning_history_list.append(hyper_tuning_history_MIMIC)
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                #random_state = 1,
                                                                )
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if n_locations == 100:
        iter_results_plot('MIMIC', max_iters, fitness_curves, running_time, input_size = n_locations)

    return fitness_curves, running_time, fitness_scores, hyper_tuning_history_list


    
#%%
def run_KColor(Tuning_flag, num_vertex, max_attempts, max_iters):
    
    print("Running K Color Randomized Optimization Experiment: \n")

    hyper_tuning_history_list = []
    
    # Define Fitness function and problem object
    p = 0.05
    g = erdos_renyi_graph(num_vertex, p)
    edges = list(g.edges)
    G = nx.Graph()
    G.add_edges_from(edges)
    plt.figure()
    nx.draw(G, with_labels=True, font_weight='bold')   # draw graph
    
    ini_state = np.random.randint(3,size=num_vertex)
    
    fitness = mlrose_hiive.MaxKColor(edges)
    problem = mlrose_hiive.DiscreteOpt(length=num_vertex, fitness_fn=fitness, maximize=False, max_val=3)
    problem.set_mimic_fast_mode(True)
     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [50, 80, 100]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, hyper_tuning_history_RHC = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_RHC)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.0001, 0.001, 0.01, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, hyper_tuning_history_SA = para_tuning('SA', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_SA)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [200], 'mutation_prob': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, hyper_tuning_history_GA = para_tuning('GA', problem, fixed_para, tunable_para, maximize=False)
        hyper_tuning_history_list.append(hyper_tuning_history_GA)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1,
                                                        )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400, 600], 'keep_pct': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, hyper_tuning_history_MIMIC = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=False) 
        hyper_tuning_history_list.append(hyper_tuning_history_MIMIC)
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1
                                                                )
    
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if num_vertex == 100:
        iter_results_plot('K-Color', max_iters, fitness_curves, running_time, input_size = num_vertex)

    return fitness_curves, running_time, fitness_scores, hyper_tuning_history_list


#%%
def run_continuous_peaks(Tuning_flag, bit_length, max_attempts, max_iters):
    print("Running Continuous Peaks Randomized Optimization Experiment: \n")

    hyper_tuning_history_list = []

    # Define Fitness function and problem object
    ini_state = np.random.randint(2,size=bit_length)
    
    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    problem = mlrose_hiive.DiscreteOpt(length=bit_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
     
    # RHC #########################################
    print(" -> Running Random Hill Climb \n")
    
    tunable_para = {'restarts': [20, 60, 100]}
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_RHC, best_fitness_value_RHC, hyper_tuning_history_RHC = para_tuning('RHC', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_RHC)
        restarts = best_param_RHC['restarts']
        
    else:    
        restarts = 100
        
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = fixed_para['max_attempts'], 
                                                            max_iters= fixed_para['max_iters'],
                                                            init_state= fixed_para ['initial_state'],
                                                            curve=True, 
                                                            random_state = 1,
                                                            restarts=restarts)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {} \n".format(rhc_time))

    # SA ############################################
    print(" -> Running Simulated Annealing \n")
    
    tunable_para = {'exp_const': [0.0001, 0.001, 0.01, 0.1], 'min_temp': [0.001, 0.01, 0.1] }
    fixed_para = {'initial_state': ini_state, 'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_SA, best_fitness_value_SA, hyper_tuning_history_SA = para_tuning('SA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_SA)
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=best_param_SA['exp_const'], min_temp=best_param_SA['min_temp'])
        
    else:    
        schedule=mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        init_state= fixed_para ['initial_state'],
                                                        curve=True, 
                                                        random_state = 1,
                                                        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {} \n".format(sa_time))
    

    # GA ############################################
    print(" -> Running Genetic Algorithm \n")
    
    tunable_para = {'pop_size': [100, 200], 'mutation_prob': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_GA, best_fitness_value_GA, hyper_tuning_history_GA = para_tuning('GA', problem, fixed_para, tunable_para, maximize=True)
        hyper_tuning_history_list.append(hyper_tuning_history_GA)
        pop_size, mutation_prob = best_param_GA['pop_size'], best_param_GA['mutation_prob']
    
    else:
        pop_size, mutation_prob = 200, 0.2
        
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts = fixed_para['max_attempts'], 
                                                        max_iters= fixed_para['max_iters'],
                                                        curve=True, 
                                                        pop_size=pop_size,
                                                        mutation_prob=mutation_prob,
                                                        random_state = 1
                                                        )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {} \n".format(ga_time))
    

    # MIMIC  #########################################
    print(" -> Running MIMIC Algorithm \n")
    
    tunable_para = {'pop_size': [200, 400, 600], 'keep_pct': [0.2, 0.4]}
    fixed_para = {'max_attempts': max_attempts,  'max_iters':max_iters}
    
    if Tuning_flag is True:
        best_param_MIMIC, best_fitness_value_MIMIC, hyper_tuning_history_MIMIC = para_tuning('MIMIC', problem, fixed_para, tunable_para, maximize=True) 
        hyper_tuning_history_list.append(hyper_tuning_history_MIMIC)
        pop_size, keep_pct = best_param_MIMIC['pop_size'], best_param_MIMIC['keep_pct']
    else:
        pop_size, keep_pct = 200, 0.25
        
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = fixed_para['max_attempts'], 
                                                                max_iters= fixed_para['max_iters'], 
                                                                curve = True, 
                                                                pop_size=pop_size,
                                                                keep_pct=keep_pct,
                                                                random_state = 1
                                                                )
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {} \n".format(mimic_time))

    fitness_curves = [rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve]
    running_time = [rhc_time, sa_time, ga_time, mimic_time]
    fitness_scores = [rhc_best_fitness, sa_best_fitness, ga_best_fitness, mimic_best_fitness]
    
    if bit_length == 200:
        iter_results_plot('Continuous Peaks', max_iters, fitness_curves, running_time, input_size = bit_length)

    return fitness_curves, running_time, fitness_scores, hyper_tuning_history_list 


#%%   
        
if __name__ == "__main__":
    
    Tuning_flag = True

  
    ### run_four_peaks
    input_size_list = [25, 50, 100, 150]
    running_time_list = []
    fitness_scores_list = []
    tuning_history_list = []
    
    for n_size in input_size_list:
        fitness_curves_fourpeak, running_time_fourpeak, fitness_scores_fourpeak, tuning_history_fourpeak = run_four_peaks(Tuning_flag, bit_length=n_size, max_attempts=200, max_iters=1000)
        running_time_list.append(running_time_fourpeak)
        fitness_scores_list.append(fitness_scores_fourpeak)
        tuning_history_list.append(tuning_history_fourpeak)
    
    input_size_plot('Four Peaks', input_size_list, fitness_scores_list, running_time_list)   
    

#    ### N-queens
#    input_size_list = [20,50,75,100]
#    running_time_list = []
#    fitness_scores_list = []
#    tuning_history_list = []
#    
#    for n_size in input_size_list:
#        fitness_curves_Nqueens, running_time_Nqueens, fitness_scores_Nqueens, tuning_history_Nqueens = Run_N_Queen(Tuning_flag, num_queens=n_size, max_attempts=100, max_iters=1000)
#        running_time_list.append(running_time_Nqueens)
#        fitness_scores_list.append(fitness_scores_Nqueens)
#        tuning_history_list.append(tuning_history_Nqueens)
#    
#    input_size_plot('N Queens', input_size_list, fitness_scores_list, running_time_list)  



    ### Flipflop
    input_size_list = [25, 50, 100, 150]
    running_time_list = []
    fitness_scores_list = []
    tuning_history_list = []

    
    for n_size in input_size_list:
        fitness_curves_FlipFlop, running_time_FlipFlop, fitness_scores_FlipFlop, tuning_history_FlipFlop = run_flipflop(Tuning_flag, bit_length=n_size, max_attempts=200, max_iters=1000)
        running_time_list.append(running_time_FlipFlop)
        fitness_scores_list.append(fitness_scores_FlipFlop)
        tuning_history_list.append(tuning_history_FlipFlop)
    
    input_size_plot('FlipFlop', input_size_list, fitness_scores_list, running_time_list) 



#    ## TSP
#    input_size_list = [50]
#    running_time_list = []
#    fitness_scores_list = []
#    tuning_history_list = []
#    
#    for n_size in input_size_list:
#        fitness_curves_TSP, running_time_TSP, fitness_scores_TSP, tuning_history_TSP = Run_TSP(Tuning_flag, n_locations=n_size, max_attempts=200, max_iters=1000)
#        running_time_list.append(running_time_TSP)
#        fitness_scores_list.append(fitness_scores_TSP)
#        tuning_history_list.append(tuning_history_TSP)
#    
#    input_size_plot('TSP', input_size_list, fitness_scores_list, running_time_list) 



    ### K-Color
    input_size_list = [200]
    running_time_list = []
    fitness_scores_list = []
    tuning_history_list = []

    
    for n_size in input_size_list:
        fitness_curves_KColor, running_time_KColor, fitness_scores_KColor, tuning_history_KColor = run_KColor(Tuning_flag, num_vertex=n_size, max_attempts=50, max_iters=1000)
        running_time_list.append(running_time_KColor)
        fitness_scores_list.append(fitness_scores_KColor)
        tuning_history_list.append(tuning_history_KColor)
    
    input_size_plot('K Color', input_size_list, fitness_scores_list, running_time_list) 
    
    
    
    ### run_contineous_peaks
    input_size_list = [50, 100, 200, 300]
    running_time_list = []
    fitness_scores_list = []
    tuning_history_list = []
    
    for n_size in input_size_list:
        fitness_curves_contpeak, running_time_contpeak, fitness_scores_contpeak, tuning_history_contpeak = run_continuous_peaks(Tuning_flag, bit_length=n_size, max_attempts=200, max_iters=1000)
        running_time_list.append(running_time_contpeak)
        fitness_scores_list.append(fitness_scores_contpeak)
        tuning_history_list.append(tuning_history_contpeak)
    
    input_size_plot('Continuous Peaks', input_size_list, fitness_scores_list, running_time_list)  
        