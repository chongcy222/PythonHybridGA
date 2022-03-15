import tkinter as tk
import numpy as np
import pygad
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from numpy import cos,sqrt,exp,pi,sin
import matplotlib.pyplot as plt
from matplotlib import cm
import time

def fitness_func_ackley(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front

    part1=-20*exp(-0.2*sqrt(((X**2)+Y**2)/2))
    C=2*pi
    part2=-exp((cos(C*X)+cos(C*Y))/2)+20+exp(1)
    total_min=part1+part2

    fitness=-total_min

    return fitness

def fitness_func_beale(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front
    min=((1.5-X+(X*Y))**2)+((2.25-X+(X*(Y**2)))**2)+((2.625-X+(X*(Y**3)))**2)
    fitness=-min

    return fitness

def fitness_func_eggholder(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front
    min=-(Y+47)*np.sin(np.sqrt(abs(X/2+(Y+47))))-X*np.sin(np.sqrt(abs(X-(Y+47))))
    fitness=-min

    return fitness

def fitness_func_griewank(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front

    part1=(X**2)/4000+((Y**2)/4000)
    part2=(cos(X/sqrt(1)))*(cos(Y/sqrt(2)))
    total_min=part1-part2+1

    fitness=-total_min

    return fitness

def fitness_func_matyas(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front
    min=0.26*((X**2)+(Y**2))-0.48*X*Y
    fitness=-min

    return fitness

def fitness_func_rastrigin(solution,solution_idx):
    Y = solution[1:]
    X = solution[:-1]
    min=(X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20 #20 is 10*dimension(2d)
    fitness=-min #change min to max problem

    return fitness

def fitness_func_rosenbrock(solution,solution_idx):
    Y = solution[1:]
    X = solution[:-1]
    min = (100.0 * (Y - X ** 2.0) ** 2.0 + (1 - X) ** 2.0)
    fitness =-min

    return fitness

def fitness_func_schwefel(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front
    min=2*418.9829-((X*sin(sqrt(abs(X))))+(Y*sin(sqrt(abs(Y)))))
    fitness=-min

    return fitness

def fitness_func_sphere(solution,solution_idx):
    Y = solution[1:] #slice front, get behind
    X = solution[:-1] #slice behind, get front
    min=(X**2)+(Y**2)
    fitness=-min

    return fitness

def ackley(x):
    Y = x[1:]
    X = x[:-1]
    part1 = -20 * exp(-0.2 * sqrt(((X ** 2) + Y ** 2) / 2))
    C = 2 * pi
    part2 = -exp((cos(C * X) + cos(C * Y)) / 2) + 20 + exp(1)
    func_value = part1 + part2
    return func_value

def beale(x):
    Y = x[1:]
    X = x[:-1]
    func_value=((1.5-X+(X*Y))**2)+((2.25-X+(X*(Y**2)))**2)+((2.625-X+(X*(Y**3)))**2)
    return func_value

def eggholder(x):
    Y = x[1:]
    X = x[:-1]
    func_value=-(Y+47)*np.sin(np.sqrt(abs(X/2+(Y+47))))-X*np.sin(np.sqrt(abs(X-(Y+47))))
    return func_value

def griewank(x):
    Y = x[1:]
    X = x[:-1]
    part1 = (X ** 2) / 4000 + ((Y ** 2) / 4000)
    part2 = (cos(X / sqrt(1))) * (cos(Y / sqrt(2)))
    func_value = part1 - part2 + 1
    return func_value

def matyas(x):
    Y = x[1:]
    X = x[:-1]
    func_value=0.26*((X**2)+(Y**2))-0.48*X*Y
    return func_value

def rastrigin(x):
    Y = x[1:]
    X = x[:-1]
    func_value = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20  # 20 is 10*dimension(2d)
    return func_value

def rosenbrock(x):
    Y = x[1:]
    X = x[:-1]
    func_value = (100.0 * (Y - X ** 2.0) ** 2.0 + (1 - X) ** 2.0)
    return func_value

def schwefel(x):
    Y = x[1:]
    X = x[:-1]
    func_value=2*418.9829-((X*sin(sqrt(abs(X))))+(Y*sin(sqrt(abs(Y)))))
    return func_value

def sphere(x):
    Y = x[1:]
    X = x[:-1]
    func_value = (X**2)+(Y**2)
    return func_value


def ackley_plot(X,Y):
    part1 = -20 * exp(-0.2 * sqrt(((X ** 2) + Y ** 2) / 2))
    C = 2 * pi
    part2 = -exp((cos(C * X) + cos(C * Y)) / 2) + 20 + exp(1)
    func_value = part1 + part2
    return func_value

def beale_plot(X,Y):
    func_value=((1.5-X+(X*Y))**2)+((2.25-X+(X*(Y**2)))**2)+((2.625-X+(X*(Y**3)))**2)
    return func_value

def eggholder_plot(X,Y):
    func_value=-(Y+47)*np.sin(np.sqrt(abs(X/2+(Y+47))))-X*np.sin(np.sqrt(abs(X-(Y+47))))
    return func_value

def griewank_plot(X,Y):
    part1 = (X ** 2) / 4000 + ((Y ** 2) / 4000)
    part2 = (cos(X/sqrt(1))) * (cos(Y/sqrt(2)))
    func_value = part1 - part2 + 1
    return func_value

def matyas_plot(X,Y):
    func_value=0.26*((X**2)+(Y**2))-0.48*X*Y
    return func_value

def rastrigin_plot(X,Y):
    func_value= (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return func_value

def rosenbrock_plot(X,Y):
    func_value = (100.0 * (Y - X ** 2.0) ** 2.0 + (1 - X) ** 2.0)
    return func_value

def schwefel_plot(X,Y):
    func_value=2*418.9829-((X*sin(sqrt(abs(X))))+(Y*sin(sqrt(abs(Y)))))
    return func_value

def sphere_plot(X,Y):
    func_value= (X**2)+(Y**2)
    return func_value

def run_preset_ackley():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Ackley"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(0, 0)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "10"
    stat_label8.configure(text=concat7)
    init_range_low = -32.768
    init_range_high = 32.768
    fitness_function = fitness_func_ackley
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): "+"BFGS"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(ackley, ga_solution, method='BFGS', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = ackley(solution)
    after_opt_answer = ackley(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-32.768, 32.768, 100)
    Y = np.linspace(-32.768, 32.768, 100)
    X, Y = np.meshgrid(X, Y)
    Z = ackley_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Ackley function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_beale():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Beale"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(3, 0.5)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "20"
    stat_label8.configure(text=concat7)
    init_range_low = -4.5
    init_range_high = 4.5
    fitness_function = fitness_func_beale
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 20
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): " + "Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(beale, ga_solution, method='Nelder-Mead', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = beale(solution)
    after_opt_answer = beale(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-4.5, 4.5, 100)
    Y = np.linspace(-4.5, 4.5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = beale_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Beale function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_eggholder():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Eggholder"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(512, 404.2319)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "-959.6407"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "200"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "60"
    stat_label8.configure(text=concat7)
    init_range_low = -512
    init_range_high = 512
    fitness_function = fitness_func_eggholder
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 200
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 60
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): " + "Basin-Hopping+Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
    opt = basinhopping(eggholder, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = eggholder(solution)
    after_opt_answer = eggholder(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-512, 512, 100)
    Y = np.linspace(-512, 512, 100)
    X, Y = np.meshgrid(X, Y)
    Z = eggholder_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Eggholder function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_griewank():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Griewank"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(0, 0)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "200"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "50"
    stat_label8.configure(text=concat7)
    init_range_low = -10
    init_range_high = 10
    fitness_function = fitness_func_griewank
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 200
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 50
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): " + "Basin-Hopping+Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
    opt = basinhopping(griewank, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = griewank(solution)
    after_opt_answer = griewank(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-10, 10, 100)
    Y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(X, Y)
    Z = griewank_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Griewank function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_matyas():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Matyas"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(0, 0)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "10"
    stat_label8.configure(text=concat7)
    init_range_low = -10
    init_range_high = 10
    fitness_function = fitness_func_matyas
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): "+"Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(matyas, ga_solution, method='Nelder-Mead', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = matyas(solution)
    after_opt_answer = matyas(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-10, 10, 100)
    Y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(X, Y)
    Z = matyas_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Matyas function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_rastrigin():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Rastrigin"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(0, 0)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "10"
    stat_label8.configure(text=concat7)
    init_range_low = -5.12
    init_range_high = 5.12
    fitness_function = fitness_func_rastrigin
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): "+"Powell"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(rastrigin, ga_solution, method='Powell', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = rastrigin(solution)
    after_opt_answer = rastrigin(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)
    Z = rastrigin_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Rastrigin function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_rosenbrock():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Rosenbrock"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(1, 1)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "10"
    stat_label8.configure(text=concat7)
    init_range_low = -2.048
    init_range_high = 2.048
    fitness_function = fitness_func_rosenbrock
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): "+"Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(rosenbrock, ga_solution, method='Nelder-Mead', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = rosenbrock(solution)
    after_opt_answer = rosenbrock(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-2.048, 2.048, 100)
    Y = np.linspace(-2.048, 2.048, 100)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Rosenbrock function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_schwefel():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Schwefel"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(420.9687, 420.9687)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "200"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "20"
    stat_label8.configure(text=concat7)
    init_range_low = -500
    init_range_high = 500
    fitness_function = fitness_func_schwefel
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 200
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 20
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): " + "Basin-Hopping+Nelder-Mead"
    hybrid_label8.configure(text=concat_hybrid_method)
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
    opt = basinhopping(schwefel, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = schwefel(solution)
    after_opt_answer = schwefel(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-500, 500, 100)
    Y = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(X, Y)
    Z = schwefel_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Schwefel function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()

def run_preset_sphere():
    start = time.time()  # assign start time of evaluation
    concat1 = "Benchmark Function: " + "Sphere"
    stat_label1.configure(text=concat1)
    concat2 = "Function solution (x0,x1): " + "(0, 0)"
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + "0"
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + "100"
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + "rank"
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + "single_point"
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + "10"
    stat_label8.configure(text=concat7)
    init_range_low = -5.12
    init_range_high = 5.12
    fitness_function = fitness_func_sphere
    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA
    np.set_printoptions(suppress=True)  # suppress small values to show 0
    # Assign GA parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions
    parent_selection_type = "rank"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    # create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    # calls the plot fitness func from pygad

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    # records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    # set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): "+"Powell"
    hybrid_label8.configure(text=concat_hybrid_method)
    opt = minimize(sphere, ga_solution, method='Powell', bounds=bounds, options={'disp': True}, tol=1e-6)
    end_hybrid = time.time()
    print("Runtime hybrid: ", end_hybrid - start)
    print("Runtime difference: ", end_hybrid - end_ga)
    opt_solution = opt.x
    before_opt_answer = sphere(solution)
    after_opt_answer = sphere(opt_solution)
    # plot the function
    fig = plt.figure()
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)
    Z = sphere_plot(X, Y)
    ax = plt.subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
    # cm.gist_heat_r
    # ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('x0 axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('Function Value')
    ax.set_title('Sphere function')

    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')

    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)

    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga - start))  # set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    plt.show()
##################################################################################################################################
def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
##################################################################################################################################
def accept_ga_para():
    start = time.time() #assign start time of evaluation
    func_name=func_input.get()
    # print(func_name)
    num_gen = int(num_generations_input.get())
    # print(num_gen)
    parent_selection=parentselection_input.get()
    # print(parent_selection)
    crossover_selection = crossover_input.get()
    # print(crossover_selection)
    mutation_rate = int(mutation_input.get())
    # print(mutation_rate)
    localsolver_name = hybrid_input.get()
    # print(localsolver_name)

    concat1="Benchmark Function: "+func_name
    stat_label1.configure(text=concat1)
    if(func_name=="Ackley"):
        func_solution="(0, 0)"
        func_answer="0"
    elif(func_name=="Beale"):
        func_solution = "(3, 0.5)"
        func_answer = "0"
    elif (func_name == "Eggholder"):
        func_solution = "(512, 404.2319)"
        func_answer = "-959.6407"
    elif (func_name == "Griewank"):
        func_solution = "(0, 0)"
        func_answer = "0"
    elif(func_name=="Matyas"):
        func_solution = "(0, 0)"
        func_answer = "0"
    elif (func_name == "Rastrigin"):
        func_solution = "(0, 0)"
        func_answer = "0"
    elif (func_name == "Rosenbrock"):
        func_solution = "(1, 1)"
        func_answer = "0"
    elif (func_name == "Schwefel"):
        func_solution = "(420.9687, 420.9687)"
        func_answer = "0"
    elif (func_name == "Sphere"):
        func_solution = "(0, 0)"
        func_answer = "0"

    concat2 = "Function solution (x0,x1): " + func_solution
    stat_label2.configure(text=concat2)
    concat3 = "Function answer f(x): " + func_answer
    stat_label3.configure(text=concat3)
    concat4 = "GA Iterations: " + str(num_gen)
    stat_label5.configure(text=concat4)
    concat5 = "GA Parent Selection Type: " + parent_selection
    stat_label6.configure(text=concat5)
    concat6 = "GA Crossover Type: " + crossover_selection
    stat_label7.configure(text=concat6)
    concat7 = "GA Mutation Percentage: " + str(mutation_rate)
    stat_label8.configure(text=concat7)


    if(func_name=="Ackley"):
        init_range_low = -32.768
        init_range_high = 32.768
        fitness_function = fitness_func_ackley
    elif(func_name=="Beale"):
        init_range_low = -4.5
        init_range_high = 4.5
        fitness_function = fitness_func_beale
    elif (func_name == "Eggholder"):
        init_range_low = -512
        init_range_high = 512
        fitness_function = fitness_func_eggholder
    elif (func_name == "Griewank"):
        init_range_low = -10
        init_range_high = 10
        fitness_function = fitness_func_griewank
    elif (func_name == "Matyas"):
        init_range_low = -10
        init_range_high = 10
        fitness_function = fitness_func_matyas
    elif (func_name == "Rastrigin"):
        init_range_low = -5.12
        init_range_high = 5.12
        fitness_function = fitness_func_rastrigin
    elif (func_name == "Rosenbrock"):
        init_range_low = -2.048
        init_range_high = 2.048
        fitness_function = fitness_func_rosenbrock
    elif (func_name == "Schwefel"):
        init_range_low = -500
        init_range_high = 500
        fitness_function = fitness_func_schwefel
    elif (func_name == "Sphere"):
        init_range_low = -5.12
        init_range_high = 5.12
        fitness_function = fitness_func_sphere

    function_inputs = np.random.uniform(init_range_low, init_range_high, 2)  # Generate 2 random starting values for GA

    np.set_printoptions(suppress=True) #suppress small values to show 0

    #Assign GA parameters
    num_generations = num_gen
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs)  # dimensions

    if(parent_selection=="roulette wheel"):
        parent_selection="rws"

    parent_selection_type = parent_selection
    keep_parents = 1
    crossover_type = crossover_selection
    mutation_type = "random"
    mutation_percent_genes = mutation_rate

    #create the GA instance and run using pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_gen)

    ga_instance.run()
    #calls the plot fitness func from pygad


    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_solution = solution
    #records the end GA time, and compare with start time
    end_ga = time.time()
    print("Runtime GA: ", end_ga - start)

    #set bounds for the benchmark function, for scipy.optimize.minimize input
    b = (init_range_low, init_range_high)
    bounds = (b, b)
    concat_hybrid_method = "Selected hybrid method (Optimizer): " + localsolver_name
    hybrid_label8.configure(text=concat_hybrid_method)

    if(localsolver_name=="Basin-Hopping+Nelder-Mead"):
        if(func_name=="Ackley"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(ackley, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = ackley(solution)
            after_opt_answer = ackley(opt_solution)


            fig = plt.figure()
            X = np.linspace(-32.768, 32.768, 100)
            Y = np.linspace(-32.768, 32.768, 100)
            X, Y = np.meshgrid(X, Y)
            Z = ackley_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Ackley function')

        elif(func_name=="Beale"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(beale, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = beale(solution)
            after_opt_answer = beale(opt_solution)

            fig = plt.figure()
            X = np.linspace(-4.5, 4.5, 100)
            Y = np.linspace(-4.5, 4.5, 100)
            X, Y = np.meshgrid(X, Y)
            Z = beale_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Beale function')

        elif (func_name == "Eggholder"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(eggholder, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = eggholder(solution)
            after_opt_answer = eggholder(opt_solution)

            fig = plt.figure()
            X = np.linspace(-4.5, 4.5, 100)
            Y = np.linspace(-4.5, 4.5, 100)
            X, Y = np.meshgrid(X, Y)
            Z = eggholder_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Eggholder function')

        elif (func_name == "Griewank"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(griewank, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = griewank(solution)
            after_opt_answer = griewank(opt_solution)

            fig = plt.figure()
            X = np.linspace(-10, 10, 100)
            Y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(X, Y)
            Z = griewank_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Griewank function')

        elif (func_name == "Matyas"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(matyas, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = matyas(solution)
            after_opt_answer = matyas(opt_solution)

            fig = plt.figure()
            X = np.linspace(-10, 10, 100)
            Y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(X, Y)
            Z = matyas_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Matyas function')

        elif (func_name == "Rastrigin"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(rastrigin, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = rastrigin(solution)
            after_opt_answer = rastrigin(opt_solution)

            fig = plt.figure()
            X = np.linspace(-5.12, 5.12, 100)
            Y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(X, Y)
            Z = rastrigin_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Rastrigin function')

        elif (func_name == "Rosenbrock"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(rosenbrock, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = rosenbrock(solution)
            after_opt_answer = rosenbrock(opt_solution)

            fig = plt.figure()
            X = np.linspace(-2.048, 2.048, 100)
            Y = np.linspace(-2.048, 2.048, 100)
            X, Y = np.meshgrid(X, Y)
            Z = rosenbrock_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Rosenbrock function')

        elif (func_name == "Schwefel"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(schwefel, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = schwefel(solution)
            after_opt_answer = schwefel(opt_solution)

            fig = plt.figure()
            X = np.linspace(-500, 500, 100)
            Y = np.linspace(-500, 500, 100)
            X, Y = np.meshgrid(X, Y)
            Z = schwefel_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Schwefel function')

        elif (func_name == "Sphere"):
            minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds}
            opt = basinhopping(sphere, ga_solution, minimizer_kwargs=minimizer_kwargs, niter=500, interval=5)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = sphere(solution)
            after_opt_answer = sphere(opt_solution)

            fig = plt.figure()
            X = np.linspace(-5.12, 5.12, 100)
            Y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(X, Y)
            Z = sphere_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Sphere function')

    else:
        if(func_name=="Ackley"):
            opt = minimize(ackley, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True},tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = ackley(solution)
            after_opt_answer = ackley(opt_solution)

            #plot the function
            fig = plt.figure()
            X = np.linspace(-32.768, 32.768, 100)
            Y = np.linspace(-32.768, 32.768, 100)
            X, Y = np.meshgrid(X, Y)
            Z = ackley_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Ackley function')

        elif(func_name=="Beale"):
            opt = minimize(beale, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True},tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = beale(solution)
            after_opt_answer = beale(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-4.5, 4.5, 100)
            Y = np.linspace(-4.5, 4.5, 100)
            X, Y = np.meshgrid(X, Y)
            Z = beale_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Beale function')

        elif (func_name == "Eggholder"):
            opt = minimize(eggholder, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = eggholder(solution)
            after_opt_answer = eggholder(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-4.5, 4.5, 100)
            Y = np.linspace(-4.5, 4.5, 100)
            X, Y = np.meshgrid(X, Y)
            Z = eggholder_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Eggholder function')

        elif (func_name == "Griewank"):
            opt = minimize(griewank, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = griewank(solution)
            after_opt_answer = griewank(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-10, 10, 100)
            Y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(X, Y)
            Z = griewank_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Griewank function')

        elif (func_name == "Matyas"):
            opt = minimize(matyas, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = matyas(solution)
            after_opt_answer = matyas(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-10, 10, 100)
            Y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(X, Y)
            Z = matyas_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Matyas function')

        elif (func_name == "Rastrigin"):
            opt = minimize(rastrigin, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = rastrigin(solution)
            after_opt_answer = rastrigin(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-5.12, 5.12, 100)
            Y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(X, Y)
            Z = rastrigin_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Rastrigin function')

        elif (func_name == "Rosenbrock"):
            opt = minimize(rosenbrock, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = rosenbrock(solution)
            after_opt_answer = rosenbrock(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-2.048, 2.048, 100)
            Y = np.linspace(-2.048, 2.048, 100)
            X, Y = np.meshgrid(X, Y)
            Z = rosenbrock_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Rosenbrock function')

        elif (func_name == "Schwefel"):
            opt = minimize(schwefel, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = schwefel(solution)
            after_opt_answer = schwefel(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-500, 500, 100)
            Y = np.linspace(-500, 500, 100)
            X, Y = np.meshgrid(X, Y)
            Z = schwefel_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Schwefel function')

        elif (func_name == "Sphere"):
            opt = minimize(sphere, ga_solution, method=localsolver_name, bounds=bounds, options={'disp': True}, tol=1e-6)
            end_hybrid = time.time()
            print("Runtime hybrid: ", end_hybrid - start)
            print("Runtime difference: ", end_hybrid - end_ga)
            opt_solution = opt.x
            before_opt_answer = sphere(solution)
            after_opt_answer = sphere(opt_solution)

            # plot the function
            fig = plt.figure()
            X = np.linspace(-5.12, 5.12, 100)
            Y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(X, Y)
            Z = sphere_plot(X, Y)
            ax = plt.subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
            fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
            # cm.gist_heat_r
            # ax.ticklabel_format(useOffset=False,style='plain')
            ax.set_xlabel('x0 axis')
            ax.set_ylabel('x1 axis')
            ax.set_zlabel('Function Value')
            ax.set_title('Sphere function')


    #plot results of GA and hybrid
    plt.figure()
    plt.contour(X, Y, Z, 500)
    final_res = opt.x
    ga_res = solution
    plt.plot(ga_res[0], ga_res[1], marker='o', markersize=10, color='b')
    plt.plot(final_res[0], final_res[1], marker='o', markersize=5, color='r')
    plt.xlabel('x0 axis')
    plt.ylabel('x1 axis')
    plt.title('Blue=GA, Red=Hybrid GA')


    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)
    print(ga_solution)
    # change labels to show GA results
    concat8 = "GA solution: " + np.array_str(ga_solution)
    ga_label1.configure(text=concat8)
    concat9 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label2.configure(text=concat9)
    concat_hybrid1 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label1.configure(text=concat_hybrid1)
    concat_hybrid2 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label2.configure(text=concat_hybrid2)



    print("---------Set precision to 4 decimal places-------------")
    np.set_printoptions(precision=4)
    print("Genetic Algorithm solution: ", ga_solution)
    print("Optimized solution: ", opt.x)
    print("Function answer f(x) before optimize: ", before_opt_answer)
    print("Function answer f(x) after optimize: ", after_opt_answer)

    concat10 = "GA solution: " + np.array_str(ga_solution)
    ga_label4.configure(text=concat10)
    concat11 = "Function answer f(x) before hybrid: " + np.array_str(before_opt_answer)
    ga_label5.configure(text=concat11)
    concat_ga_runtime = "GA Runtime (seconds): " + str("{:.4f}".format(end_ga-start)) #set the time to 2 decimal places
    ga_label6.configure(text=concat_ga_runtime)
    concat_hybrid3 = "Solution after hybrid: " + np.array_str(opt_solution)
    hybrid_label4.configure(text=concat_hybrid3)
    concat_hybrid4 = "Function answer f(x) after hybrid: " + np.array_str(after_opt_answer)
    hybrid_label5.configure(text=concat_hybrid4)
    concat_hybrid_runtime = "Hybrid+GA Runtime (seconds): " + str("{:.4f}".format(end_hybrid - start))
    hybrid_label6.configure(text=concat_hybrid_runtime)
    concat_runtime_diff = "Hybrid and GA Runtime Difference (seconds): " + str("{:.4f}".format(end_hybrid - end_ga))
    hybrid_label7.configure(text=concat_runtime_diff)

    ga_instance.plot_fitness()
    
    plt.show()


#####################################################################################################################
window=tk.Tk()
window.title("Hybrid Genetic Algorithm Evaluation")
label = tk.Label(text="Select GA run options manually: ", font=(None, 10),bg='#bbed55')
label.grid(row=0,column=0,padx=8,pady=8,sticky='w')
window.wm_geometry("1366x650")

func_label = tk.Label(text="Benchmark Function:")
func_label.grid(row=1,column=0,padx=5,pady=5,sticky='w')
func_input= tk.StringVar(window)
func_input.set("Ackley") #Default benchmark function name
gen_dropdown=tk.OptionMenu(window,func_input,"Ackley","Beale","Eggholder","Griewank","Matyas","Rastrigin","Rosenbrock","Schwefel","Sphere")
gen_dropdown.grid(row=2,column=0,padx=5,pady=5,sticky='w')

numgen_label = tk.Label(text="Number of Generations:")
numgen_label.grid(row=3,column=0,padx=5,pady=5,sticky='w')
num_generations_input= tk.StringVar(window)
num_generations_input.set("100") #Default number of generations
gen_dropdown=tk.OptionMenu(window,num_generations_input,"100","200","300","400","500")
gen_dropdown.grid(row=4,column=0,padx=5,pady=5,sticky='w')

parentselection_label = tk.Label(text="Parent Selection Type:")
parentselection_label.grid(row=5,column=0,padx=5,pady=5,sticky='w')
parentselection_input= tk.StringVar(window)
parentselection_input.set("rank") #Default number of generations
parentselection_dropdown=tk.OptionMenu(window,parentselection_input,"rank","tournament","roulette wheel")
parentselection_dropdown.grid(row=6,column=0,padx=5,pady=5,sticky='w')

crossover_label = tk.Label(text="Crossover Type:")
crossover_label.grid(row=7,column=0,padx=5,pady=5,sticky='w')
crossover_input= tk.StringVar(window)
crossover_input.set("single_point") #Default number of generations
crossover_dropdown=tk.OptionMenu(window,crossover_input,"single_point","two_points")
crossover_dropdown.grid(row=8,column=0,padx=5,pady=5,sticky='w')

mutation_label = tk.Label(text="Mutation Method: Random")
mutation_label.grid(row=9,column=0,padx=5,pady=5,sticky='w')
mutation_label = tk.Label(text="Mutation Percentage:")
mutation_label.grid(row=10,column=0,padx=5,pady=5,sticky='w')
mutation_input= tk.StringVar(window)
mutation_input.set("10") #Default number of generations
mutation_dropdown=tk.OptionMenu(window,mutation_input,"10","20","30","40","50","60","70","80","90","100")
mutation_dropdown.grid(row=11,column=0,padx=5,pady=5,sticky='w')

hybrid_label = tk.Label(text="Hybrid Method (Optimizer):")
hybrid_label.grid(row=12,column=0,padx=5,pady=5,sticky='w')
hybrid_input= tk.StringVar(window)
hybrid_input.set("Nelder-Mead") #Default benchmark function name
hybrid_dropdown=tk.OptionMenu(window,hybrid_input,"Nelder-Mead","Basin-Hopping+Nelder-Mead","BFGS","Powell")
hybrid_dropdown.grid(row=13,column=0,padx=5,pady=5,sticky='w')

inputgen_button = tk.Button(text='Run',width=10,command=accept_ga_para,bg='#ebb9b9')
inputgen_button.grid(row=14,column=0,padx=8,pady=13,sticky='w')
#####################################################################################################################
label2 = tk.Label(text="Or Run with Presets ", font=(None, 10),bg='#e5eb7f')
label2.grid(row=1,column=1,padx=8,pady=8,sticky='w')
ackley_button = tk.Button(text='Ackley Preset',command=run_preset_ackley)
ackley_button.grid(row=2,column=1,padx=8,pady=8,sticky='w')
beale_button = tk.Button(text='Beale Preset',command=run_preset_beale)
beale_button.grid(row=3,column=1,padx=8,pady=8,sticky='w')
eggholder_button = tk.Button(text='Eggholder Preset',command=run_preset_eggholder)
eggholder_button.grid(row=4,column=1,padx=8,pady=8,sticky='w')
griewank_button = tk.Button(text='Griewank Preset',command=run_preset_griewank)
griewank_button.grid(row=5,column=1,padx=8,pady=8,sticky='w')
matyas_button = tk.Button(text='Matyas Preset',command=run_preset_matyas)
matyas_button.grid(row=6,column=1,padx=8,pady=8,sticky='w')
rastrigin_button = tk.Button(text='Rastrigin Preset',command=run_preset_rastrigin)
rastrigin_button.grid(row=7,column=1,padx=8,pady=8,sticky='w')
rosenbrock_button = tk.Button(text='Rosenbrock Preset',command=run_preset_rosenbrock)
rosenbrock_button.grid(row=8,column=1,padx=8,pady=8,sticky='w')
schwefel_button = tk.Button(text='Schwefel Preset',command=run_preset_schwefel)
schwefel_button.grid(row=9,column=1,padx=8,pady=8,sticky='w')
sphere_button = tk.Button(text='Sphere Preset',command=run_preset_sphere)
sphere_button.grid(row=10,column=1,padx=8,pady=8,sticky='w')
#####################################################################################################################
label3 = tk.Label(text="-----Function Info-----",font=(None, 10),bg='#4af0ed')
label3.grid(row=1,column=2)
stat_label1=tk.Label(text="Benchmark Function: ")
stat_label1.grid(row=2,column=2,sticky='w')
stat_label2=tk.Label(text="Function solution (x0,x1): ")
stat_label2.grid(row=3,column=2,sticky='w')
stat_label3=tk.Label(text="Function answer f(x): ")
stat_label3.grid(row=4,column=2,sticky='w')
stat_label4=tk.Label(text="-----Genetic Algorithm Parameters-----", font=(None, 10),bg='#c9f73e')
stat_label4.grid(row=5,column=2)
stat_label5=tk.Label(text="Number of Generations/GA Iterations: ")
stat_label5.grid(row=6,column=2,sticky='w')
stat_label6=tk.Label(text="GA Parent Selection Type: ")
stat_label6.grid(row=7,column=2,sticky='w')
stat_label7=tk.Label(text="GA Crossover Type: ")
stat_label7.grid(row=8,column=2,sticky='w')
stat_label8=tk.Label(text="GA Mutation Percentage: ")
stat_label8.grid(row=9,column=2,sticky='w')
note_label01=tk.Label(text="-----Note-----",bg='#e8d1eb')
note_label01.grid(row=11,column=2)
note_label1=tk.Label(text="All benchmark functions are minimization problems.")
note_label1.grid(row=12,column=2,sticky='w')
note_label2=tk.Label(text="f(x) value is best when smallest or close to 0.")
note_label2.grid(row=13,column=2,sticky='w')
note_label6=tk.Label(text="PLEASE RE-RUN A FEW TIMES IF SOLUTION IS BAD(STUCKED).",bg='#fae49b')
note_label6.grid(row=14,column=2,sticky='w')

#####################################################################################################################
label4=tk.Label(text="-----Genetic Algorithm Outputs-----", font=(None, 10),bg='#c9f73e')
label4.grid(row=1,column=3)
ga_label1=tk.Label(text="GA solution: ")
ga_label1.grid(row=2,column=3,sticky='w')
ga_label2=tk.Label(text="Function answer f(x) before hybrid: ")
ga_label2.grid(row=3,column=3,sticky='w')
ga_label3=tk.Label(text="-----Set Precision to 4 decimal places-----",bg='#f0c45d')
ga_label3.grid(row=4,column=3,sticky='w')
ga_label4=tk.Label(text="GA solution: ")
ga_label4.grid(row=5,column=3,sticky='w')
ga_label5=tk.Label(text="Function answer f(x) before hybrid: ")
ga_label5.grid(row=6,column=3,sticky='w')
ga_label6=tk.Label(text="GA Runtime (seconds): ",bg='#f06e5d')
ga_label6.grid(row=7,column=3,sticky='w')
note_label02=tk.Label(text="-----Note-----",bg='#e8d1eb')
note_label02.grid(row=11,column=3)
note_label3=tk.Label(text="  When hybrid solution is similar to GA,")
note_label3.grid(row=12,column=3,sticky='w')
note_label4=tk.Label(text="  it is most likely that it stucked in a valley,")
note_label4.grid(row=13,column=3,sticky='w')
note_label5=tk.Label(text="  local minima (which is not the smallest yet).")
note_label5.grid(row=14,column=3,sticky='w')

#####################################################################################################################
label5=tk.Label(text="-----Optimized(Hybrid) Outputs-----", font=(None, 10),bg='#ed8aea')
label5.grid(row=1,column=4)
hybrid_label1=tk.Label(text="Solution after hybrid: ")
hybrid_label1.grid(row=2,column=4,sticky='w')
hybrid_label2=tk.Label(text="Function answer f(x) after hybrid: ")
hybrid_label2.grid(row=3,column=4,sticky='w')
hybrid_label3=tk.Label(text="-----Set Precision to 4 decimal places-----",bg='#f0c45d')
hybrid_label3.grid(row=4,column=4,sticky='w')
hybrid_label4=tk.Label(text="Solution after hybrid: ")
hybrid_label4.grid(row=5,column=4,sticky='w')
hybrid_label5=tk.Label(text="Function answer f(x) after hybrid: ")
hybrid_label5.grid(row=6,column=4,sticky='w')
hybrid_label6=tk.Label(text="Hybrid+GA Runtime (seconds): ",bg='#f06e5d')
hybrid_label6.grid(row=7,column=4,sticky='w')
hybrid_label7=tk.Label(text="Hybrid and GA Runtime Difference (seconds): ",bg='#f06e5d')
hybrid_label7.grid(row=8,column=4,sticky='w')
hybrid_label8=tk.Label(text="Selected hybrid method (Optimizer): ")
hybrid_label8.grid(row=9,column=4,sticky='w')


window.mainloop()