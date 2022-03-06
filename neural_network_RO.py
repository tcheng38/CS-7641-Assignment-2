# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:41:28 2022

@author: cheng164
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.exceptions import ConvergenceWarning
from  warnings import simplefilter

import mlrose_hiive
import time



def data_load(file):
    
    ## Data Loading and Visualization
    df=pd.read_csv(file)
    df.head()
    
    #describing the data
    print(df.info())
    df.describe()
    
    #encoding the Gender attribute
    plt.figure()
    sns.countplot(df.Gender)
    df['Gender'].replace({'Male':1,'Female':0},inplace=True)
    df['Dataset'].replace(2,0, inplace=True)
    # let's look on target variable - classes imbalanced?
    df['Dataset'].value_counts()
    plt.figure()
    sns.countplot(df.Dataset)
    
    
    #checking for missing values as per column
    df.isna().sum()
    
    #checking the rows with the missing values
    df[df['Albumin_and_Globulin_Ratio'].isna()]
    df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())
    
    # Explore correlations visually
    f, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    
    #  Data preprossesing
     
    X = df.drop(['Dataset'], axis=1)
    y = df['Dataset']
     
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
    
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    ## Define scoring metric depending on it's binary or multiclass classification problem
    if y_test.nunique()>2:   # multiclass case
        scoring_metric = 'f1_macro' 
    else:
        scoring_metric = 'balanced_accuracy' 
    
    return X_train, X_test, y_train, y_test, scoring_metric



def run_NeuralNet_RHC(algorithm, X_train, X_test, y_train, y_test, scoring_metric, learning_rate, restarts):  
    
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [40], activation ='relu', 
                                 algorithm = algorithm, 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = learning_rate, restarts = restarts,
                                 early_stopping = False, clip_max = 5, max_attempts = 100, random_state = 1, curve = True)

    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    fitting_time = end_time-start_time
    
    # Predict labels for train and test set, and then calculate accuracy score
    y_pred_train = nn_model.predict(X_train)
    y_pred_test = nn_model.predict(X_test)
    
    if scoring_metric == 'balanced_accuracy':
        balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
        
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred_test))
    fitness_curve = nn_model.fitness_curve
    loss= nn_model.loss 
    
    print('Algorithm {} has train accuracy= {} and test accuracy= {}'.format(algorithm, balanced_accuracy_train, balanced_accuracy_test))
    print('Classification Report: \n', classification_report(y_test, y_pred_test))   
    return balanced_accuracy_train, balanced_accuracy_test, fitting_time, fitness_curve , loss 



def run_NeuralNet_SA(algorithm, X_train, X_test, y_train, y_test, scoring_metric, learning_rate, schedule):  
    
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [40], activation ='relu', 
                                 algorithm = algorithm, 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = learning_rate, schedule = schedule,
                                 early_stopping = False, clip_max = 5, max_attempts = 100, random_state = 1, curve = True)

    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    fitting_time = end_time-start_time
    
    # Predict labels for train and test set, and then calculate accuracy score
    y_pred_train = nn_model.predict(X_train)
    y_pred_test = nn_model.predict(X_test)
    
    if scoring_metric == 'balanced_accuracy':
        balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
        
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred_test))
    fitness_curve = nn_model.fitness_curve
    loss= nn_model.loss 
    
    print('Algorithm {} has train accuracy= {} and test accuracy= {}'.format(algorithm, balanced_accuracy_train, balanced_accuracy_test))
    print('Classification Report: \n', classification_report(y_test, y_pred_test))   
    return balanced_accuracy_train, balanced_accuracy_test, fitting_time, fitness_curve , loss 




def run_NeuralNet_GA(algorithm, X_train, X_test, y_train, y_test, scoring_metric, learning_rate, pop_size, mutation_prob):  
    
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [40], activation ='relu', 
                                 algorithm = algorithm, 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = learning_rate, pop_size = pop_size, mutation_prob = mutation_prob,
                                 early_stopping = False, clip_max = 5, max_attempts = 100, curve = True)

    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    fitting_time = end_time-start_time
    
    # Predict labels for train and test set, and then calculate accuracy score
    y_pred_train = nn_model.predict(X_train)
    y_pred_test = nn_model.predict(X_test)
    
    if scoring_metric == 'balanced_accuracy':
        balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
        
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred_test))
    fitness_curve = nn_model.fitness_curve
    loss= nn_model.loss 
    
    print('Algorithm {} has train accuracy= {} and test accuracy= {}'.format(algorithm, balanced_accuracy_train, balanced_accuracy_test))
    print('Classification Report: \n', classification_report(y_test, y_pred_test))   
    return balanced_accuracy_train, balanced_accuracy_test, fitting_time, fitness_curve , loss 



def run_NeuralNet_GD(algorithm, X_train, X_test, y_train, y_test, scoring_metric, learning_rate):  
    
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [40], activation ='relu', 
                                 algorithm = algorithm, 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = learning_rate,
                                 early_stopping = False, clip_max = 5, max_attempts = 100, random_state = 1, curve = True)

    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    fitting_time = end_time-start_time
    
    # Predict labels for train and test set, and then calculate accuracy score
    y_pred_train = nn_model.predict(X_train)
    y_pred_test = nn_model.predict(X_test)
    
    if scoring_metric == 'balanced_accuracy':
        balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
        
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred_test))
    fitness_curve = nn_model.fitness_curve
    loss= nn_model.loss 
    
    print('Algorithm {} has train accuracy= {} and test accuracy= {}'.format(algorithm, balanced_accuracy_train, balanced_accuracy_test))
    print('Classification Report: \n', classification_report(y_test, y_pred_test))   
    return balanced_accuracy_train, balanced_accuracy_test, fitting_time, fitness_curve , loss 



if __name__ == "__main__":

    iteration_nums = 1000
    algorithm_list = ['gradient_descent', 'random_hill_climb', 'simulated_annealing', 'genetic_alg']
    X_train, X_test, y_train, y_test, scoring_metric = data_load("indian_liver_patient.csv")
    
    accuracy_train_RHC, accuracy_test_RHC, fitting_time_RHC, fitness_curve_RHC, loss_RHC = run_NeuralNet_RHC('random_hill_climb', X_train, X_test, y_train, y_test, scoring_metric, learning_rate = 0.1, restarts=10)
    accuracy_train_SA, accuracy_test_SA, fitting_time_SA, fitness_curve_SA, loss_SA = run_NeuralNet_SA('simulated_annealing', X_train, X_test, y_train, y_test, scoring_metric, learning_rate = 0.1, schedule = mlrose_hiive.GeomDecay())
    accuracy_train_GA, accuracy_test_GA, fitting_time_GA, fitness_curve_GA, loss_GA = run_NeuralNet_GA('genetic_alg', X_train, X_test, y_train, y_test, scoring_metric, learning_rate = 0.001, pop_size = 500, mutation_prob=0.2)
    accuracy_train_GD, accuracy_test_GD, fitting_time_GD, fitness_curve_GD, loss_GD = run_NeuralNet_GD('gradient_descent', X_train, X_test, y_train, y_test, scoring_metric, learning_rate = 0.001)

    plt.figure()
    iterations = range(1, iteration_nums+1)
    plt.plot(iterations, fitness_curve_RHC[0:iteration_nums,0], label='RHC', color='green')
    plt.plot(iterations, fitness_curve_SA[0:iteration_nums,0], label='SA', color='red')
    plt.plot(iterations, fitness_curve_GA[0:iteration_nums,0], label='GA', color='blue')
    plt.plot(iterations, -1*fitness_curve_GD[0:iteration_nums], label='GD', color='black')

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")    
    plt.legend(loc="best")
    plt.title('Fitness Curves for Neural Network weight optimization')
    plt.savefig("results/NN_fitness_curve.png")
    
    