import random
import os

class GeneticAlgorithm:

    def __init__(self, population, tournament_size=5, elitism=True, elite_size=1):
        self.population = population
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.elite_size = elite_size
        self.generation = 1

    def run(self, max_generations):
        print("Starting genetic algorithm...\n")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):
        
        if os.path.exists('./fittest.txt'):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        
        with open('./fittest.txt', append_write) as f:
            
            print("Evaluating generation #{}...\n\n".format(self.generation))
            f.write("Evaluating generation #{}...\n\n".format(self.generation))
            fittest = self.population.get_fittest()[0]
            print("Fittest individual is: \n")
            f.write("Fittest individual is: \n{}".format(fittest.genes))
            print(fittest.genes)
            print("\nFitness value is: {}\n".format(round(fittest.fitness, 4)))
            f.write("\nFitness value is: {}\n".format(round(fittest.fitness, 4)))
            
        new_population = self.population.__class__(self.population.species, indiv_list=[], maximize=self.population.maximize)
        if self.elitism:
            [new_population.add_individual(i) for i in self.population.get_fittest(self.elite_size)]
        while new_population.population_size < self.population.population_size:
            child = self.tournament_select().reproduce(self.tournament_select())
            child.mutate()
            new_population.add_individual(child)
        self.population = new_population

    def tournament_select(self):
        tournament = self.population.__class__(
            self.population.species, indiv_list=[
                self.population[i] for i in random.sample(range(self.population.population_size), self.tournament_size)
            ], maximize=self.population.maximize
        )
        fittest = tournament.get_fittest()[0]
        
        return fittest
        

if __name__ == '__main__':
    pass