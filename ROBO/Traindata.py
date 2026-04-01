
import torch
import cocoex
import random
import numpy as np

def train_data_problem(fun, dimension, n_samples, value_range ):
    fun = int(fun)
    dimension = int(dimension)
    n_samples = int(n_samples)
   
    # Setze Seed für Reproduzierbarkeit
    seed = 42
    random.seed(seed)       
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    # Suite laden 
    suite = cocoex.Suite("bbob", "", "")                        # Hier den Sting dynamisch Anpassen oder kompletten suite laden 
    problem = suite.get_problem_by_function_dimension_instance(
    function= fun,              # z. B. f9
    dimension= dimension,       # gewünschte Dimension
    instance=1                  # Instanz
    )

    print(f"Problem: {problem.name}, Dimension: ")


    # Erzeugung der Trainingsdaten X
    train_X = torch.rand(n_samples, problem.dimension) * (value_range[1]- value_range[0]) + value_range[0]


    # Funktionswerte berechen

    train_Y = []
    for x in train_X:
        x_numpy = x.numpy()
        y_val = problem(x_numpy)
        train_Y.append(y_val)
        print(f"f({x_numpy}) = {y_val:.6f}")
    


    train_Y = torch.tensor(train_Y, dtype=torch.float32).unsqueeze(-1)
    train_X = train_X.double()
    train_Y = train_Y.double()

    return(train_X,train_Y)


def evaluate_candidate(trained_acquisition_X,fun, n_samples, value_range, dimension):
    
    # Setze Seed für Reproduzierbarkeit
    seed = 42
    random.seed(seed)       
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    # Suite laden 
    suite = cocoex.Suite("bbob", "", "")                        # Hier den Sting dynamisch Anpassen oder kompletten suite laden 
    problem = suite.get_problem_by_function_dimension_instance(
    function= fun,              # z. B. f9
    dimension= dimension,       # gewünschte Dimension
    instance=1                  # Instanz
    )
    acquisition_Y = []
    for x in trained_acquisition_X:
        x_numpy = x.numpy()
        y_val = problem(x_numpy)
        acquisition_Y.append(y_val)
        print(f"f({x_numpy}) = {y_val:.6f}")
    


    acquisition_Y = torch.tensor(acquisition_Y, dtype=torch.float32).unsqueeze(-1)
    trained_acquisition_X = trained_acquisition_X.double()
    acquisition_Y = acquisition_Y.double()

    return(trained_acquisition_X,acquisition_Y)


