
class Population():
    
    def __init__(self, species, size, crossover_rate=0.5, mutation_rate=0.015, maximize=True):
        self.species = species
        self.maximize = maximize
        self.population_size = size
        self.individuals = [self.species.random_init(crossover_rate, mutation_rate) for _ in range(size)]
        print("Initializing a random population. Size: {}".format(size))

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_fittest(self):
        for gen in self.individuals:
            if gen.fitness == None:
                gen.set_fitness()
        if self.maximize:
            return max(gen.fitness for gen in self.individuals)
        return min(gen.fitness for gen in self.individuals)

    def __getitem__(self, item):
        return self.individuals[item]

if __name__ == '__main__':
    pass