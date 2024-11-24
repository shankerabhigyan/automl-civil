import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import random
from copy import deepcopy

# Import the model classes
from src.anfis_sa import ANFISPredictor
from src.xgb import CustomXGBoost
from src.logistic import LogisticNAS
from src.svm import SVMNAS

class ModelGene:
    def __init__(self, model_type: str, params: Dict[str, Any]):
        self.model_type = model_type
        self.params = params
        self.fitness = None
        self.model = None
    
    def create_model(self, target_column: str):
        if self.model_type == 'anfis':
            self.model = ANFISPredictor(
                target_column=target_column,
                hidden_dims=self.params['hidden_dims'],
                use_attention=self.params['use_attention'],
                attention_dim=self.params['attention_dim'],
                num_rules=self.params['num_rules'],
                dropout_rate=self.params['dropout_rate']
            )
        elif self.model_type == 'xgboost':
            self.model = CustomXGBoost(
                target_column=target_column,
                scale_features=self.params['scale_features'],
                scale_target=self.params['scale_target'],
                xgb_params=self.params['xgb_params'],
                validation_size=self.params['validation_size'],
                early_stopping_rounds=self.params['early_stopping_rounds']
            )
        elif self.model_type == 'logistic':
            self.model = LogisticNAS(
                target_column=target_column,
                use_attention=self.params['use_attention'],
                attention_dim=self.params['attention_dim'],
                use_feature_selection=self.params['use_feature_selection'],
                regularization=self.params['regularization'],
                temperature=self.params['temperature']
            )
        elif self.model_type == 'svm':
            self.model = SVMNAS(
                target_column=target_column,
                kernel_type=self.params['kernel_type'],
                kernel_params=self.params['kernel_params'],
                use_feature_selection=self.params['use_feature_selection'],
                temperature=self.params['temperature'],
                C=self.params['C']
            )
        return self.model

class GeneticOptimizer:
    def __init__(self, 
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_rate: float = 0.2,
                 elite_size: int = 2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_gene = None
        self.generation_history = []

    def generate_random_params(self, model_type: str) -> Dict[str, Any]:
        if model_type == 'anfis':
            return {
                'hidden_dims': [
                    random.choice([64, 128, 256, 512]) 
                    for _ in range(random.randint(2, 4))
                ],
                'use_attention': random.choice([True, False]),
                'attention_dim': random.choice([16, 32, 64, 128]),
                'num_rules': random.randint(3, 10),
                'dropout_rate': random.uniform(0.0, 0.5)
            }
        elif model_type == 'xgboost':
            return {
                'scale_features': True,
                'scale_target': True,
                'xgb_params': {
                    'max_depth': random.randint(3, 10),
                    'learning_rate': random.uniform(0.01, 0.3),
                    'n_estimators': random.randint(50, 300),
                    'min_child_weight': random.randint(1, 7),
                    'subsample': random.uniform(0.6, 1.0),
                    'colsample_bytree': random.uniform(0.6, 1.0)
                },
                'validation_size': 0.2,
                'early_stopping_rounds': random.randint(20, 50)
            }
        elif model_type == 'logistic':
            return {
                'use_attention': random.choice([True, False]),
                'attention_dim': random.choice([16, 32, 64, 128]),
                'use_feature_selection': random.choice([True, False]),
                'regularization': random.choice(['l1', 'l2']),
                'temperature': random.uniform(0.5, 2.0)
            }
        else:  # svm
            return {
                'kernel_type': random.choice(['rbf', 'poly', 'linear']),
                'kernel_params': {
                    'gamma': random.uniform(0.1, 5.0),
                    'degree': random.randint(2, 5),
                    'coef0': random.uniform(0.0, 2.0)
                },
                'use_feature_selection': random.choice([True, False]),
                'temperature': random.uniform(0.5, 2.0),
                'C': random.uniform(0.1, 10.0)
            }

    def initialize_population(self) -> List[ModelGene]:
        population = []
        model_types = ['anfis', 'xgboost']
        anfiscount = 0
        xgboostcount = 0
        
        for _ in range(self.population_size):
            model_type = random.choice(model_types)
            params = self.generate_random_params(model_type)
            population.append(ModelGene(model_type, params))
            if model_type == 'anfis':
                anfiscount += 1
            else:
                xgboostcount += 1

        print(f"Initial Population: {anfiscount} ANFIS, {xgboostcount} XGBoost")
                
            
        return population

    def crossover(self, parent1: ModelGene, parent2: ModelGene) -> ModelGene:
        # If same model type, mix parameters
        if parent1.model_type == parent2.model_type:
            new_params = {}
            for key in parent1.params:
                if isinstance(parent1.params[key], dict):
                    new_params[key] = {}
                    for subkey in parent1.params[key]:
                        new_params[key][subkey] = random.choice(
                            [parent1.params[key][subkey], parent2.params[key][subkey]]
                        )
                else:
                    new_params[key] = random.choice(
                        [parent1.params[key], parent2.params[key]]
                    )
            return ModelGene(parent1.model_type, new_params)
        # If different model types, return the better parent
        else:
            return deepcopy(parent1 if parent1.fitness > parent2.fitness else parent2)

    def mutate(self, gene: ModelGene) -> ModelGene:
        if random.random() < self.mutation_rate:
            # Randomly modify some parameters
            new_params = deepcopy(gene.params)
            random_params = self.generate_random_params(gene.model_type)
            
            # Randomly choose parameters to mutate
            for key in new_params:
                if isinstance(new_params[key], dict):
                    for subkey in new_params[key]:
                        if random.random() < self.mutation_rate:
                            new_params[key][subkey] = random_params[key][subkey]
                else:
                    if random.random() < self.mutation_rate:
                        new_params[key] = random_params[key]
            
            return ModelGene(gene.model_type, new_params)
        return gene

    def evaluate_population(self, population: List[ModelGene], 
                          data: pd.DataFrame, target_column: str) -> List[float]:
        for gene in population:
            if gene.fitness is None:
                try:
                    model = gene.create_model(target_column)
                    model.fit(data)
                    gene.fitness = model.get_fitness_score(data)
                except Exception as e:
                    print(f"Error training {gene.model_type} model: {str(e)}")
                    gene.fitness = 0.0
        
        return [gene.fitness for gene in population]

    def select_parents(self, population: List[ModelGene], 
                      fitnesses: List[float]) -> List[ModelGene]:
        # Tournament selection
        tournament_size = 3
        selected = []
        
        # Keep elite individuals
        sorted_population = [x for _, x in sorted(
            zip(fitnesses, population), 
            key=lambda pair: pair[0],
            reverse=True
        )]
        selected.extend(sorted_population[:self.elite_size])
        
        # Tournament selection for the rest
        while len(selected) < self.population_size:
            tournament = random.sample(list(enumerate(population)), tournament_size)
            winner = max(tournament, key=lambda x: fitnesses[x[0]])[1]
            selected.append(deepcopy(winner))
            
        return selected

    def optimize(self, data: pd.DataFrame, target_column: str) -> Tuple[ModelGene, List[float]]:
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate current population
            fitnesses = self.evaluate_population(population, data, target_column)
            
            # Store best model and fitness
            best_idx = np.argmax(fitnesses)
            if self.best_gene is None or fitnesses[best_idx] > self.best_gene.fitness:
                self.best_gene = deepcopy(population[best_idx])
            
            self.generation_history.append({
                'generation': generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'best_model_type': population[best_idx].model_type
            })
            
            print(f"Best Fitness: {max(fitnesses):.4f}")
            print(f"Average Fitness: {np.mean(fitnesses):.4f}")
            print(f"Best Model Type: {population[best_idx].model_type}")
            
            # Selection
            parents = self.select_parents(population, fitnesses)
            
            # Create new population through crossover and mutation
            new_population = []
            new_population.extend(parents[:self.elite_size])  # Keep elite individuals
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return self.best_gene, self.generation_history

def plot_optimization_history(history: List[Dict]) -> None:
    import matplotlib.pyplot as plt
    
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, label='Best Fitness', marker='o')
    plt.plot(generations, avg_fitness, label='Average Fitness', marker='s')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    data = pd.read_csv("./data/concrete/cleaned_concrete.csv")
    target_column = "csMPa"
    
    optimizer = GeneticOptimizer(
        population_size=20,
        generations=50,
        mutation_rate=0.2,
        elite_size=2
    )
    
    best_model, history = optimizer.optimize(data, target_column)
    
    print("\nBest Model Found:")
    print(f"Model Type: {best_model.model_type}")
    print(f"Parameters: {best_model.params}")
    print(f"Fitness Score: {best_model.fitness:.4f}")
    
    # Plot optimization progress
    plot_optimization_history(history)