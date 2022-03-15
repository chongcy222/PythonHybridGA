# PythonHybridGA
My final year project: Evaluating Hybrid Genetic Algorithms

The traditional Genetic Algorithm does not often deliver the best result, as the algorithm is great at searching around huge solution spaces but not in a specific local area. In order to prove the significance of hybrid, a traditional Genetic Algorithm model is implemented along with a few hybrid models. The performance of them will be evaluated with different benchmarking problems, such as the Rastrigin and Rosenbrock function. The project is related to the field of Genetic Algorithm and Optimization.

For instance, a problem's final answer could be x coordinate at (0,0) as we are doing a minimization problem. However the Genetic Algorithm can only reach -1 instead of 0, which is very close. With hybrid, we aim to achieve 0. 

The genetic algorithm is implemented using "PyGAD".
While hybrid (local search algorithms) were implemented using "SciPy".
