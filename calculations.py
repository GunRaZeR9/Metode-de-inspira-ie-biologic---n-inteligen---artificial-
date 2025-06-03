#====================== IMPORTS ======================
import pandas
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import random


#====================== DATA LOADING AND ANALYSIS ======================
def load_data():
    """
    Load data from the CSV file.
    
    Returns:
    -------
    pandas.DataFrame:
        The loaded dataset
    """
    FILE_PATH = "data/data.csv"
    df = pandas.read_csv(FILE_PATH)
    return df


def get_public_orgs():
    """
    Count the number of public organizations in the dataset.
    
    Returns:
    -------
    int:
        Number of public organizations
    """
    df = load_data()
    public_df = df[df["Public?"] == 1]
    num = len(public_df)
    print("There are ", num, "public organizations.")
    return num


def revenue_per_industry():
    """
    Calculate the average revenue per industry.
    
    Returns:
    -------
    pandas.Series:
        Series containing average revenue for each industry
    """
    df = load_data()
    revenue_by_industry = df.groupby("Industry")["Revenue"].sum()
    count_by_industry = df["Industry"].value_counts()
    rev_ratio = revenue_by_industry / count_by_industry
    return rev_ratio


def highest_revenue_industry():
    """
    Find the industry with the highest total revenue.
    
    Returns:
    -------
    str:
        Name of the industry with highest revenue
    """
    df = load_data()
    revenue_by_industry = df.groupby("Industry")["Revenue"].sum()
    top_revenue_industry = revenue_by_industry.idxmax()
    return top_revenue_industry


#====================== GA COMPONENTS ======================
def load_covtype_data():
    """
    Load the covtype dataset and split into train and test sets.
    
    Returns:
    -------
    tuple:
        X_train, X_test, y_train, y_test
    """
    print("Loading covtype dataset...")
    X, y = fetch_covtype(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(individual, X, y):
    """
    Evaluate a model with given hyperparameters using cross-validation.
    
    Parameters:
    ----------
    individual : list
        List of hyperparameters [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    X : array-like
        Feature matrix
    y : array-like
        Target vector
        
    Returns:
    -------
    tuple:
        Single-element tuple containing the mean cross-validation accuracy
    """
    n_estimators = individual[0]
    max_depth = individual[1]
    min_samples_split = individual[2]
    min_samples_leaf = individual[3]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean(),
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return 0.0,


def setup_ga(X_train, y_train):
    """
    Set up the genetic algorithm toolbox for hyperparameter optimization.
    
    Parameters:
    ----------
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training target vector
        
    Returns:
    -------
    deap.base.Toolbox:
        Configured GA toolbox
    """
    # Clean up any previous DEAP definitions to avoid errors on re-runs
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individual' in creator.__dict__:
        del creator.Individual
        
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define genes
    toolbox.register("attr_n_estimators", random.randint, 10, 200)
    toolbox.register("attr_max_depth", random.randint, 1, 30)
    toolbox.register("attr_min_samples_split", random.randint, 2, 20)
    toolbox.register("attr_min_samples_leaf", random.randint, 1, 10)
    
    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_estimators, toolbox.attr_max_depth,
                      toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_model, X=X_train, y=y_train)
    
    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 
                    low=[10, 1, 2, 1], 
                    up=[200, 30, 20, 10], 
                    indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox


#====================== OPTIMIZATION FUNCTIONS ======================
def run_optimization(pop_size=10, num_generations=5, verbose=True):
    """
    Run genetic algorithm optimization for hyperparameter tuning.
    
    Parameters:
    ----------
    pop_size : int
        Population size for GA
    num_generations : int
        Number of generations to evolve
    verbose : bool
        Whether to print progress information and plot results
        
    Returns:
    -------
    dict:
        Dictionary containing optimization results and metrics
    """
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = load_covtype_data()
    
    # Use a smaller sample for faster demonstration
    sample_size = 10000
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    if verbose:
        print(f"Setting up genetic algorithm with {pop_size} individuals and {num_generations} generations...")
    toolbox = setup_ga(X_train_sample, y_train_sample)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Hall of fame to keep track of best individuals
    hof = tools.HallOfFame(1)
    
    # Statistics to track progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the algorithm
    if verbose:
        print("Running genetic algorithm optimization...")
    population, log = algorithms.eaSimple(population, toolbox, 
                                         cxpb=0.7,  # Crossover probability
                                         mutpb=0.3,  # Mutation probability
                                         ngen=num_generations, 
                                         stats=stats,
                                         halloffame=hof,
                                         verbose=verbose)
    
    # Get the best hyperparameters
    best_hyperparams = hof[0]
    
    if verbose:
        print("\nBest hyperparameters found:")
        print(f"n_estimators: {best_hyperparams[0]}")
        print(f"max_depth: {best_hyperparams[1]}")
        print(f"min_samples_split: {best_hyperparams[2]}")
        print(f"min_samples_leaf: {best_hyperparams[3]}")
    
    # Create model with best hyperparameters
    best_model = RandomForestClassifier(
        n_estimators=best_hyperparams[0],
        max_depth=best_hyperparams[1],
        min_samples_split=best_hyperparams[2],
        min_samples_leaf=best_hyperparams[3],
        random_state=42,
        n_jobs=-1
    )
    
    # Train on full training set
    if verbose:
        print("Training best model on full training dataset...")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print("\nTest accuracy with best hyperparameters:", accuracy)
    
    # Create a baseline model with default hyperparameters
    baseline_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    if verbose:
        print("Training baseline model for comparison...")
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    
    if verbose:
        print("\nBaseline model (default hyperparameters) accuracy:", baseline_accuracy)
        print(f"Improvement: {accuracy - baseline_accuracy:.4f} ({(accuracy - baseline_accuracy) / baseline_accuracy * 100:.2f}%)")
    
    end_time = time.time()
    runtime = end_time - start_time
    
    if verbose:
        print(f"\nTotal optimization time: {runtime:.2f} seconds")
        
        # Plot the results
        plot_optimization_results(log, accuracy, baseline_accuracy, best_hyperparams)
    
    # Return comprehensive results
    results = {
        'log': log,
        'best_hyperparams': best_hyperparams,
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': accuracy - baseline_accuracy,
        'improvement_percent': (accuracy - baseline_accuracy) / baseline_accuracy * 100,
        'runtime': runtime,
        'pop_size': pop_size,
        'num_generations': num_generations,
        'final_max_fitness': log.select('max')[-1] if len(log.select('max')) > 0 else 0,
        'final_avg_fitness': log.select('avg')[-1] if len(log.select('avg')) > 0 else 0
    }
    
    return results


def run_brute_force_comparison():
    """
    Run a brute force comparison of different GA parameter combinations.
    
    Returns:
    -------
    list:
        List of dictionaries containing results for each parameter combination
    """
    print("\n" + "="*50)
    print("RUNNING BRUTE FORCE COMPARISON OF GA PARAMETERS")
    print("="*50)
    
    # Define the ranges to explore
    pop_sizes = [5, 8, 10, 12, 15]
    generations = [5, 10, 15]
    
    # Store results
    results = []
    
    # Get a baseline model accuracy once to use for all comparisons
    X_train, X_test, y_train, y_test = load_covtype_data()
    baseline_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    print("Training baseline model for comparison...")
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
    
    # Total number of combinations to try
    total_combinations = len(pop_sizes) * len(generations)
    current_combination = 0
    
    # Try each combination
    for pop_size in pop_sizes:
        for num_gen in generations:
            current_combination += 1
            print(f"\nRunning combination {current_combination}/{total_combinations}: Population={pop_size}, Generations={num_gen}")
            
            # Run the optimization with these parameters
            result = run_optimization(pop_size=pop_size, num_generations=num_gen, verbose=False)
            result['baseline_accuracy'] = baseline_accuracy  # Use the same baseline for fair comparison
            results.append(result)
            
            # Print a brief summary
            print(f"  Accuracy: {result['accuracy']:.4f}, Improvement: {result['improvement']:.4f}, Runtime: {result['runtime']:.2f}s")
    
    # Visualize the comparison results
    visualize_brute_force_results(results, baseline_accuracy)
    
    # Find the best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nBEST CONFIGURATION:")
    print(f"Population Size: {best_result['pop_size']}")
    print(f"Number of Generations: {best_result['num_generations']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Improvement over baseline: {best_result['improvement']:.4f} ({best_result['improvement_percent']:.2f}%)")
    print(f"Runtime: {best_result['runtime']:.2f} seconds")
    
    return results


#====================== VISUALIZATION FUNCTIONS ======================
def plot_optimization_results(log, accuracy, baseline_accuracy, best_hyperparams):
    """
    Plot the optimization progress and comparison with baseline.
    
    Parameters:
    ----------
    log : deap.tools.Logbook
        Log of the GA evolution
    accuracy : float
        Accuracy of the best model
    baseline_accuracy : float
        Accuracy of the baseline model
    best_hyperparams : list
        Best hyperparameters found
    """
    gen = log.select("gen")
    max_fitness = log.select("max")
    avg_fitness = log.select("avg")
    min_fitness = log.select("min")
    std_fitness = log.select("std")

    plt.figure(figsize=(12, 6))
    plt.plot(gen, max_fitness, label="Max Fitness", marker='o')
    plt.plot(gen, avg_fitness, label="Avg Fitness", marker='o')
    plt.plot(gen, min_fitness, label="Min Fitness", marker='o')
    plt.fill_between(gen, np.array(avg_fitness) - np.array(std_fitness), 
                     np.array(avg_fitness) + np.array(std_fitness), alpha=0.2, label="Std Dev")
    plt.title("GA Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ga_optimization_progress.png")
    print("GA optimization progress plot saved to 'ga_optimization_progress.png'")
    plt.show()

    # Bar plot for accuracy comparison
    plt.figure(figsize=(6, 5))
    plt.bar(["Baseline", "GA Best"], [baseline_accuracy, accuracy], color=["gray", "blue"])
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate([baseline_accuracy, accuracy]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("ga_vs_baseline_accuracy.png")
    print("GA vs Baseline accuracy plot saved to 'ga_vs_baseline_accuracy.png'")
    plt.show()


def visualize_brute_force_results(results, baseline_accuracy):
    """
    Visualize the results from brute force parameter comparison.
    
    Parameters:
    ----------
    results : list
        List of dictionaries containing results for each parameter combination
    baseline_accuracy : float
        Accuracy of the baseline model
    """
    # Organize data by population size and generations
    pop_sizes = sorted(list(set(r['pop_size'] for r in results)))
    generations = sorted(list(set(r['num_generations'] for r in results)))
    
    # Create a grid for accuracy values
    accuracy_grid = np.zeros((len(pop_sizes), len(generations)))
    runtime_grid = np.zeros((len(pop_sizes), len(generations)))
    improvement_grid = np.zeros((len(pop_sizes), len(generations)))
    
    # Fill the grids
    for r in results:
        pop_idx = pop_sizes.index(r['pop_size'])
        gen_idx = generations.index(r['num_generations'])
        accuracy_grid[pop_idx, gen_idx] = r['accuracy']
        runtime_grid[pop_idx, gen_idx] = r['runtime']
        improvement_grid[pop_idx, gen_idx] = r['improvement']
    
    #----- Accuracy Heatmap and Parameter Analysis -----
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Heatmap of accuracy
    plt.subplot(2, 2, 1)
    im = plt.imshow(accuracy_grid, cmap='viridis')
    plt.colorbar(im, label='Accuracy')
    plt.title('Model Accuracy by GA Parameters')
    plt.xlabel('Number of Generations')
    plt.ylabel('Population Size')
    plt.xticks(range(len(generations)), generations)
    plt.yticks(range(len(pop_sizes)), pop_sizes)
    
    # Add text annotations to the heatmap
    for i in range(len(pop_sizes)):
        for j in range(len(generations)):
            plt.text(j, i, f"{accuracy_grid[i, j]:.4f}", 
                    ha="center", va="center", color="w", fontweight='bold')
    
    # Plot 2: Accuracy vs Population Size for different generation counts
    plt.subplot(2, 2, 2)
    for i, gen in enumerate(generations):
        gen_results = [r for r in results if r['num_generations'] == gen]
        gen_results = sorted(gen_results, key=lambda x: x['pop_size'])
        
        plt.plot([r['pop_size'] for r in gen_results], 
                [r['accuracy'] for r in gen_results], 
                marker='o', label=f'Gen={gen}')
    
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')
    plt.title('Accuracy vs Population Size')
    plt.xlabel('Population Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Accuracy vs Generation Count for different population sizes
    plt.subplot(2, 2, 3)
    for i, pop in enumerate(pop_sizes):
        pop_results = [r for r in results if r['pop_size'] == pop]
        pop_results = sorted(pop_results, key=lambda x: x['num_generations'])
        
        plt.plot([r['num_generations'] for r in pop_results], 
                [r['accuracy'] for r in pop_results], 
                marker='o', label=f'Pop={pop}')
    
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')
    plt.title('Accuracy vs Number of Generations')
    plt.xlabel('Number of Generations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Runtime comparison
    plt.subplot(2, 2, 4)
    x = range(len(results))
    plt.bar(x, [r['runtime'] for r in results])
    plt.xticks(x, [f"P{r['pop_size']}-G{r['num_generations']}" for r in results], rotation=45)
    plt.title('Runtime by Configuration')
    plt.xlabel('Configuration (Population-Generations)')
    plt.ylabel('Runtime (seconds)')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('ga_parameter_comparison.png')
    print("\nBrute force comparison visualization saved to 'ga_parameter_comparison.png'")
    plt.show()
    
    #----- Improvement Percentages -----
    plt.figure(figsize=(10, 6))
    configs = [f"P{r['pop_size']}-G{r['num_generations']}" for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    # Sort by improvement
    sorted_indices = np.argsort(improvements)
    sorted_configs = [configs[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_configs, sorted_improvements, color='purple')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f"{sorted_improvements[i]:.2f}%",
                ha='center', va='bottom', rotation=45)
    
    plt.title('Improvement Percentage Over Baseline by Configuration')
    plt.xlabel('Configuration (Population-Generations)')
    plt.ylabel('Improvement (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ga_improvement_comparison.png')
    print("Improvement comparison visualization saved to 'ga_improvement_comparison.png'")
    plt.show()
    
    #----- Hyperparameter Analysis -----
    plt.figure(figsize=(15, 10))
    
    # Extract values for each hyperparameter
    n_estimators = [r['best_hyperparams'][0] for r in results]
    max_depths = [r['best_hyperparams'][1] for r in results]
    min_samples_splits = [r['best_hyperparams'][2] for r in results]
    min_samples_leafs = [r['best_hyperparams'][3] for r in results]
    
    # Create 4 subplots for each hyperparameter
    plt.subplot(2, 2, 1)
    plt.bar(configs, n_estimators, color='blue')
    plt.title('Best n_estimators by Configuration')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.bar(configs, max_depths, color='green')
    plt.title('Best max_depth by Configuration')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.bar(configs, min_samples_splits, color='red')
    plt.title('Best min_samples_split by Configuration')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.bar(configs, min_samples_leafs, color='purple')
    plt.title('Best min_samples_leaf by Configuration')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ga_hyperparameters_by_config.png')
    print("Hyperparameter comparison by configuration saved to 'ga_hyperparameters_by_config.png'")
    plt.show()


#====================== MAIN EXECUTION ======================
def main():
    """
    Main function to run the analysis and optimization.
    """
    # Run original calculations
    print("Running original calculations...")
    try:
        num_public_orgs = get_public_orgs()
        rev_per_industry = revenue_per_industry()
        highest_rev_industry = highest_revenue_industry()

        print("Number of public organizations:", num_public_orgs)
        print("Revenue per industry:")
        print(rev_per_industry)
        print("Industry with the highest revenue:", highest_rev_industry)
    except Exception as e:
        print(f"Error running original calculations: {e}")
        print("Continuing with GA optimization...")
    
    # Options for GA optimization
    print("\n\n" + "="*50)
    print("GENETIC ALGORITHM OPTIMIZATION OPTIONS")
    print("="*50)
    print("1. Run single GA optimization")
    print("2. Run brute force comparison of GA parameters")
    print("3. Skip GA optimization")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        pop_size = int(input("Enter population size (default 10): ") or 10)
        num_generations = int(input("Enter number of generations (default 5): ") or 5)
        run_optimization(pop_size=pop_size, num_generations=num_generations)
    elif choice == "2":
        run_brute_force_comparison()
    else:
        print("GA optimization skipped.")


if __name__ == "__main__":
    main()