import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.position = np.random.uniform(self.lower_bound, self.upper_bound)
        self.best = self.position.copy()
        self.best_value = -inf
        self.speed = np.random.uniform(-(self.upper_bound-self.lower_bound), (self.upper_bound-self.lower_bound))


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hyperparams = hyperparams
        self.num_particles = hyperparams.num_particles
        self.particles = np.empty(hyperparams.num_particles, dtype=Particle)
        for i in range(hyperparams.num_particles):
            self.particles[i] = Particle(self.lower_bound, self.upper_bound)
        self.particle_counter = 0
        # self.best_particle = self.particles[0]
        self.best_iteration_position = None
        self.best_iteration_value = -inf
        self.best_global_position = self.particles[0].position
        self.best_global_value = -inf

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        # return self.best_particle.position # Remove this line
        return self.best_global_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        # return self.best_particle.best_value  # Remove this line
        return self.best_global_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        rp = random.uniform(0, 1)
        rg = random.uniform(0, 1)

        self.particles[self.particle_counter].speed = self.particles[self.particle_counter].speed * self.hyperparams.inertia_weight
        self.particles[self.particle_counter].speed = self.particles[self.particle_counter].speed + rp * self.hyperparams.cognitive_parameter * (self.particles[self.particle_counter].best - self.particles[self.particle_counter].position)
        self.particles[self.particle_counter].speed = self.particles[self.particle_counter].speed + rg * self.hyperparams.social_parameter * (self.get_best_position() - self.particles[self.particle_counter].position)

        self.particles[self.particle_counter].speed = np.minimum(np.maximum(self.particles[self.particle_counter].speed, -(self.upper_bound - self.lower_bound)), (self.upper_bound - self.lower_bound))

        self.particles[self.particle_counter].position = self.particles[self.particle_counter].position + self.particles[self.particle_counter].speed

        self.particles[self.particle_counter].position = np.minimum(np.maximum(self.particles[self.particle_counter].position, self.lower_bound), self.upper_bound)


        return self.particles[self.particle_counter].position

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # Todo: implement
        if self.best_iteration_value > self.best_global_value:
            self.best_global_position = self.best_iteration_position
            self.best_global_value = self.best_iteration_value

        self.particle_counter = 0

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        if value > self.particles[self.particle_counter].best_value:
            self.particles[self.particle_counter].best = self.particles[self.particle_counter].position.copy()
            self.particles[self.particle_counter].best_value = value
        if value > self.best_iteration_value:
            self.best_iteration_value = value
            self.best_iteration_position = self.particles[self.particle_counter].position.copy()
            # if value > self.get_best_value():
                # self.best_particle = self.particles[self.particle_counter]

        self.particle_counter += 1
        if self.particle_counter == self.num_particles:
            self.advance_generation()

