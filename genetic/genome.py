import random
from collections import defaultdict

def random_log_uniform(minimum, maximum, base, eps=1e-12):
    """Generate a random number which is uniform in a
    logarithmic scale. If base > 0 scale goes from minimum
    to maximum, if base < 0 vice versa, and if base is 0,
    use a uniform scale.
    """
    if base == 0:
        return random.uniform(minimum, maximum)
    minimum += eps  # Avoid math domain error when minimum is zero
    if base > 0:
        return base ** random.uniform(math.log(minimum, base), math.log(maximum, base))
    base = abs(base)
    return maximum - base ** random.uniform(math.log(eps, base), math.log(maximum - minimum, base))


class Genome:

    def __init__(self, genome, genes, crossover_rate, mutation_rate):
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown

    def validate_genome(self):
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")
        for gene in self.genome:
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")

    def validate_genes(self):
        """Check that genes are compatible with genome."""
        if set(self.genome.keys()) != set(self.__class__.gene_set):
            raise ValueError("Genes passed don't correspond to individual's genome.")
    
    def reproduce(self, partner):
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.genes.items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.genes[name]
            else:
                child_genes[name] = value
        return self.__class__(
            self.genome, child_genes, self.crossover_rate, self.mutation_rate,
        )

    def crossover(self, partner):
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.get_genes().keys():
            if random.random() < self.crossover_rate:
                self.genes[name], partner.genes[name] = partner.genes[name], self.genes[name]
                self.set_fitness(None)
                partner.set_fitness(None)

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.set_fitness(None)  # The mutation produces a new individual

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.genome, self.genes.copy(), self.crossover_rate,
            self.mutation_rate, **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)
        return individual_copy


class A2CGenome(Genome):
    
    gene_set = ('gamma', 'alpha', 'beta', 'hidden_dim')
    
    def __init__(self):
        super().__init__()