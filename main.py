# -*- coding: utf-8 -*-
import quandl
import random
from live_trading_algorithms import Algorithm, algo, btest
from live_trading_algorithms.Algorithm import Algorithm
from live_trading_algorithms.universe import Universe
from backtesting import static_papertrading
from backtesting.Individual import Individual
import pandas as pd
from random import randint
import sys
import uuid
import math
import matplotlib.pyplot as plt

def mutation(child):
    print("mutation")
    mutated_gene = randint(0, chromosome_length)
    if (mutated_gene == 0):
        child.stock = Universe[randint(0, len(Universe))]
    if (mutated_gene == 1):
        child.short_window = randint(10,50)
    if (mutated_gene == 2):
        child.long_window = randint(30,100)
    if (mutated_gene == 3):  
        child.trading_rule = "SMA"
    if (mutated_gene == 4):
        child.buy_quantity = randint(10,100)
    return child

def crossover(parents, eliminations, population):
    #long and short window are crossed over from the same parent some of the time    
    child1 = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
    child1.sharpe_ratio = 0.0
    child2 = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
    child2.sharpe_ratio = 0.0  
    population.append(child1)
    population.append(child2)
    crossover_point = randint(0, chromosome_length)
    print("crossover point:", crossover_point)

    #parent1 loop
    for i in range(0, crossover_point):
        if (i == 0):
            child1.stock = parents[0].stock
            child2.stock = parents[1].stock
        if (i == 1):
            child1.short_window = parents[0].short_window
            child2.short_window = parents[1].short_window
        if (i == 2):
            child1.long_window = parents[0].long_window
            child2.long_window = parents[1].long_window
        if (i == 3):
            child1.trading_rule = parents[0].trading_rule
            child2.trading_rule = parents[1].trading_rule
        if (i == 4):
            child1.buy_quantity = parents[0].buy_quantity
            child2.buy_quantity = parents[1].buy_quantity

    #parent2 loop
    for i in range(crossover_point, chromosome_length):
        if (i == 0):
            child1.stock = parents[1].stock
            child2.stock = parents[0].stock
        if (i == 1):
            child1.short_window = parents[1].short_window
            child2.short_window = parents[0].short_window
        if (i == 2):
            child1.long_window = parents[1].long_window
            child2.long_window = parents[0].long_window
        if (i == 3):
            child1.trading_rule = parents[1].trading_rule
            child2.trading_rule = parents[0].trading_rule
        if (i == 4):
            child1.buy_quantity = parents[1].buy_quantity
            child2.buy_quantity = parents[0].buy_quantity

    #first index of population is changed here
    child1.name = str(uuid.uuid4())[0:10] + "_" + child1.stock
    child2.name = str(uuid.uuid4())[0:10] + "_" + child2.stock
    print("crossover function")
    print(parents[0].name, parents[0].trading_rule, parents[0].stock, parents[0].short_window, parents[0].long_window, parents[0].sharpe_ratio)
    print(parents[1].name, parents[1].trading_rule, parents[1].stock, parents[1].short_window, parents[1].long_window, parents[1].sharpe_ratio)
    print(child1.name, child1.trading_rule, child1.stock, child1.short_window, child1.long_window)
    print(child2.name, child2.trading_rule, child2.stock, child2.short_window, child2.long_window)
    print("eliminations")
    print(eliminations[0].name, eliminations[0].sharpe_ratio)
    print(eliminations[1].name, eliminations[1].sharpe_ratio)

    if randint(1,100) <= mutation_probability:
        child1 = mutation(child1)

    #steady-state regeneration of population
    population.append(parents[0])
    population.append(parents[1])

    return population

def fitness(population):
    #fitness variables
    highest_sharpe_ratio = 0.0
    highest_absolute_return = 0.0
    highest_alpha = 0.0
    lowest_sharpe_ratio = 0.0
    lowest_absolute_return = 0.0
    lowest_alpha = 0.0
    fittest_individuals = list()
    weakest_individuals = list()
    '''superimpose security time series against portfolio with trading signals'''
    '''evolution involves scrapping strategy entirely or modifying parameters'''
    '''testing for best fitness value'''
    '''1:1 fitness value to evolution for comparison'''
    '''evolutions could be fixed on specific time periods i.e. 1 month per evolution of individuals to preserve scarce data points'''

    for i in range(0,2):
        fittest_individual = population[0]
        weakest_individual = population[0]
        highest_sharpe_ratio = 0
        lowest_sharpe_ratio = 0
        for individual in population:
            if individual.sharpe_ratio > highest_sharpe_ratio:
                fittest_individual = individual
                highest_sharpe_ratio = individual.sharpe_ratio
            if individual.sharpe_ratio < lowest_sharpe_ratio:
                weakest_individual = individual
                lowest_sharpe_ratio = individual.sharpe_ratio
        #add to lists for crossover and removal from list (steady-state)
        fittest_individuals.append(fittest_individual)
        weakest_individuals.append(weakest_individual)
        #remove from population
        if fittest_individual in population:
            population.remove(fittest_individual)
            fittest_individuals_list.append(fittest_individual)
        if weakest_individual in population:
            population.remove(weakest_individual)
    population = crossover(fittest_individuals, weakest_individuals, population)
    return population

    '''
    for i in range(0,2):
        fittest_individual = population[0]
        for individual in population:
            if individual.alpha > highest_alpha:
                fittest_individual = individual
                highest_alpha = individual.alpha
        if fittest_individual in population:
            population.remove(fittest_individual)
        fittest_individuals.append(fittest_individual)

    crossover(fittest_individuals, population)
    fittest_individuals_list.append(fittest_individual)
    print("The fittest individual of the population is " + fittest_individual.name + " with an alpha of " + str(fittest_individual.alpha))
    return fittest_individual
    '''

'''steady state model: remove two worst and replace with two best'''
'''generational model: new population'''

if __name__ == '__main__':

    # parameters: strategy, stock universe, start date, end date, cash
    # parameterize algorithm
    # backtesting better for accuracy optimization
    # mutation - move operator: changing thing in isolation
    # instantiate population
    quandl.ApiConfig.api_key = "HAQ5HX1UH9eB9virjnGF "
    file_log = open("logs.txt", "w")
    # instantiate and execute individual algorithms
    # assemble stocklists
    #global variables
    #potential 2D array of start/end time periods
    #multidimensional Universe array to specificy asset classes

    #changing this isn't a good idea due to evolutions going into a timeframe in the future
    starting_period = pd.Timestamp('2013-1-1')
    ending_period = pd.Timestamp('2013-6-1')
    #explain why the timeperiod moves with each generation
    #fail to allocate bitmap for large evolutions > 5
    #for final plot graphing correlation between evolutions and improved fitness value
    evolvedhighestfitnessvalue = dict()
    evolvedaveragefitnessvalue = dict()
    population = list()
    stocklist = list()
    fittest_individuals_list = list()
    Universe_Equity = list()
    Universe_Commodities = ['ICE_RS2', 'CME_CL11', 'ICE_CC2', 'ICE_M2', 'ICE_RS2', 'ICE_G2', 'ICE_KC2', 'ICE_T6']
    #adjustable variables
    fitness_values = ['sharpe_ratio', 'alpha', 'absolute_return', 'historical_average_return']
    fitness_value = 'sharpe_ratio'
    chromosome_length = 5
    evolutions = 10
    mutation_probability = 5
    population_size = 20
    single_stock_optimization = False
    dataset = 'CHRIS'

    if single_stock_optimization:
        Universe_Equity = Universe[randint(0,len(Universe))]
    else:
        for i in range(0,population_size):
            Universe_Equity.append(Universe[randint(0,len(Universe))])

    if dataset == 'CHRIS':
        Universe_Equity = Universe_Commodities

    #individual generation
    #change len(Universe_Equity) to population_size if interested in single stock optimization
    for i in range(0, len(Universe_Equity)):
        #populate stocklist rather than single stock trading. very very important for making crossovers more exciting
        '''
        maxInt = random.randint(0, len(Universe_Equity))
        for i in range(0, maxInt):
            stocklist.append(Universe_Equity[random.randint(0, maxInt)])
        '''
        s1 = Individual(Universe_Equity[i], randint(10,50), randint(50,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
        population.append(s1)
        s1.main()

        #removal of anomalies in sharpe ratios
        #some Individuals did not have valid sharpe ratios and therefore are discarded as they cannot be evolved or participate in the fitness landscape
        #only commented out for commodity testing
        '''
        if (math.isnan(s1.sharpe_ratio)):
            Universe.remove(s1.stock)
            population.remove(s1)
            s1 = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA')
            population.append(s1)
            s1.main()
        '''

    print("initial population: ")
    file_log.write("initial population: \n")
    for i in population:
        print(i.name, i.sharpe_ratio)
        file_log.write(i.name + " " + str(i.sharpe_ratio) + "\n")            
        
    for j in range(0, evolutions):  
        #some fitness values can be negative so initializing at zero would fail to document them in the post-fitness if statements 
        for i in population:
            i.main()
            if (math.isnan(i.sharpe_ratio)):
                population.remove(i)
                i = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
                population.append(i)
                i.main()

        print("pre-fitness")
        file_log.write("pre-fitness, generation number " + str(j) + "\n")
        for i in population:
            print(i.name, i.sharpe_ratio)
            file_log.write(i.name + " " + str(i.sharpe_ratio) + "\n")    
            #exception handlers
            if (math.isnan(i.sharpe_ratio)):
                population.remove(i)
                i = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
                population.append(i)
                i.main()
       
        population = fitness(population)

        #post-fitness analysis of population
        highest_fitness_value = -100
        average_fitness_value = -100
        print("post-fitness")
        file_log.write("post-fitness, generation number " + str(j) + "\n")
        for i in population:
            #catch child individuals with unassigned fitness values before final population printout
            '''
            if i.sharpe_ratio == 0.0:
                i.main()
            if (math.isnan(i.sharpe_ratio)):
                Universe.remove(i.stock)
                population.remove(i)
                i = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA')
                population.append(i)
                i.main()
            '''
            #computing highest and average for each generation seperated by fitness value in question

            if fitness_value is "sharpe_ratio":
                if i.sharpe_ratio > highest_fitness_value:
                    highest_fitness_value = i.sharpe_ratio
                average_fitness_value += i.sharpe_ratio
            if fitness_value is "alpha":
                if i.alpha > highest_fitness_value:
                    highest_fitness_value = i.alpha
                average_fitness_value += i.alpha
            if fitness_value is "absolute_return":
                if i.absolute_return > highest_fitness_value:
                    highest_fitness_value = i.absolute_return
                average_fitness_value += i.absolute_return
            if fitness_value is "historical_average_return":
                if i.historical_average_return > highest_fitness_value:
                    highest_fitness_value = i.historical_average_return
                average_fitness_value += i.historical_average_return
            
            print(i.name, i.sharpe_ratio, i.start, i.end)
            file_log.write(i.name + " " + str(i.sharpe_ratio) + "\n")
            i.start += pd.Timedelta('180 day')
            i.end += pd.Timedelta('180 day')
        average_fitness_value = average_fitness_value / len(population)
        evolvedhighestfitnessvalue.update({j: highest_fitness_value})
        evolvedaveragefitnessvalue.update({j: average_fitness_value})

    for i in population:
        i.main()
        if (math.isnan(i.sharpe_ratio)):
            population.remove(i)
            i = Individual(Universe[randint(0,len(Universe))], randint(10,50), randint(30,100), starting_period, ending_period, dataset, 'SMA', randint(10,100))
            population.append(i)
            i.main()

    print("final population:")
    file_log.write("final population: \n")
    for i in population:
        print(i.name, i.sharpe_ratio, i.long_window, i.short_window, i.start, i.end)
        file_log.write("Name: " + i.name + "Sharpe Ratio: " + str(i.sharpe_ratio) + "Long Window: " + str(i.long_window) + "Short Window: " + str(i.short_window) + "Start Date: " + str(i.start) + "End Date: " + str(i.end) + "Historical Returns: " + str(i.historical_returns))

    print("evolutionary progress")
    file_log.write("evolutionary progress \n")
    for i in evolvedhighestfitnessvalue:
        print(i, evolvedhighestfitnessvalue[i])

    for i in evolvedaveragefitnessvalue:
        print(i, evolvedaveragefitnessvalue[i])

    file_log.write("evolved highest fitness value \n")
    file_log.write(str(evolvedhighestfitnessvalue) + "\n")
    file_log.write("evolved average fitness value \n")
    file_log.write(str(evolvedaveragefitnessvalue))
    file_log.close()
    
    values = list(evolvedhighestfitnessvalue.values())
    keys = list(evolvedhighestfitnessvalue.keys())
    fig, ax = plt.subplots()
    ax.plot(keys, values)
    ax.set(xlabel='generation number', ylabel='fitness value',
       title="highest " + str(fitness_value) + ", population: " + str(population_size) + ", mutation probability: " + str(mutation_probability) + ", evolutions: " + str(evolutions))
    ax.grid()
    fig.savefig("highest.png")
    plt.show()

    values = list(evolvedaveragefitnessvalue.values())
    keys = list(evolvedaveragefitnessvalue.keys())
    fig, ax = plt.subplots()
    ax.plot(keys, values)
    ax.set(xlabel='generation number', ylabel='fitness value',
       title="average " + str(fitness_value) + ", population: " + str(population_size) + ", mutation probability: " + str(mutation_probability) + ", evolutions: " + str(evolutions))
    ax.grid()
    fig.savefig("averages.png")
    plt.show()

    #adjust population list
    #del population[:]

    '''
    for evolution, individual in enumerate(fittest_individuals_list):
        print(evolution, individual.name, str(individual.sharpe_ratio), str(individual.alpha))
    '''

    '''
    evolving population
    selection - tournament, roulette
    gene crossovers and mutations
    multi-objective evolutionary algorithms
    pareto front
    mutiple different fitness functions
    weighted aggregate value
    learning horizons
    number of parameters
    '''

    '''conflict between individual genes and fitness landscape constraints'''
    '''old stuff'''
    #QUANTCONNECT_API_KEY = '9ca3fa20fe44d977c3a49388f90687b0'
    #QUANTCONNECT_API_ID = '66576'
    #Alg1 = Algorithm('America/New York','SMA','2017-11-07','2017-12-10',stocklist,len(stocklist),100000,3)
    #population.append(Alg1)
    #Alg1.main()
    #algo.main()
    # testing
    # btest.simulate()
    #static_papertrading.main('AAPL', 40, 100)
    #static_papertrading.main('MSFT', 40, 100)
    # evolute population
    # evaluates each algorithm