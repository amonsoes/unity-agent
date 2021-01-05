import argparse

from genetic.tournament import GeneticAlgorithm
from genetic.genome import A2CGenome, PPOGenome
from genetic.population import Population

from a2c_cartpole import utils

def find_best_hyperparams(species, 
                          population_size, 
                          crossover_rate, 
                          mutation_rate, 
                          elitism, 
                          elite_size, 
                          maximize, 
                          generation_epochs):
    
    if species in ('a2c', 'A2C', 'Actor-Critic'):
        population = Population(A2CGenome, 
                size=population_size, 
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                maximize=maximize)

    elif species in ('ppo', 'PPO', 'Proximal Policy Optimization'):
        population = Population(PPOGenome, 
                size=population_size, 
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                maximize=maximize)
    else:
        assert argparse.ArgumentError('WRONG INPUT FOR SPECIES. SET TO PPO OR A2C')
        population = None
        
        
    selection = GeneticAlgorithm(population, elitism=elitism, elite_size=elite_size)
    selection.run(generation_epochs)
    with open('./best_hyperparams.txt', 'w') as w:
        w.write(selection.population.get_fittest().genes)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('species', type=str, help='set species')
    parser.add_argument('--pop_size', type=int, default=50, help='set population rate')
    parser.add_argument('--cross_rate', type=float, default=0.3, help='set crossover rate')
    parser.add_argument('--mut_rate', type=float, default=0.015, help='set mutation rate')
    parser.add_argument('--elitism', type=utils.str2bool, default=True, help='set if best indivs get to live on')
    parser.add_argument('--elite_size', type=int, default=3, help='set how many best indivs get to live on')
    parser.add_argument('--maximize', type=utils.str2bool, default=True, help='should indivs be maximized or minimized')
    parser.add_argument('--gen_epochs', type=int, default=7, help='number of generations')
    
    args = parser.parse_args()
    
    find_best_hyperparams(species=args.species, 
                          population_size=args.pop_size, 
                          crossover_rate=args.cross_rate, 
                          mutation_rate=args.mut_rate, 
                          elitism=args.elitism, 
                          elite_size=args.elite_size, 
                          maximize=args.maximize, 
                          generation_epochs=args.gen_epochs)