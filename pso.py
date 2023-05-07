import random
import numpy as np
import matplotlip.pyplot as plt

class PSO:
    def __init__(self, objective_function, param_ranges, num_particles=10, max_iter=50, c1=2, c2=2, w=0.7):
        self.objective_function = objective_function
        self.param_ranges = param_ranges
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.logs = []

    def run(self):
        # Initialize the particles and their velocities
        particles = np.array([self._generate_particle() for i in range(self.num_particles)])
        velocities = np.zeros((self.num_particles, len(self.param_ranges)))
        best_particle_positions = particles.copy()
        best_particle_scores = np.full(self.num_particles, -np.inf)
        global_best_particle_position = None
        global_best_particle_score = -np.inf

        # Iterate for the maximum number of iterations
        for i in range(self.max_iter):
            # Evaluate the objective function for each particle
            particle_scores = np.array([self.objective_function(particle) for particle in particles])
            self.logs.append(particle_scores)

            # Update the best positions and scores for each particle
            for j in range(self.num_particles):
                if particle_scores[j] > best_particle_scores[j]:
                    best_particle_positions[j] = particles[j]
                    best_particle_scores[j] = particle_scores[j]
                if particle_scores[j] > global_best_particle_score:
                    global_best_particle_position = particles[j]
                    global_best_particle_score = particle_scores[j]

            # Update the velocities and positions of the particles
            for j in range(self.num_particles):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                velocities[j] = self.w * velocities[j] \
                                 + self.c1 * r1 * (best_particle_positions[j] - particles[j]) \
                                 + self.c2 * r2 * (global_best_particle_position - particles[j])
                particles[j] += velocities[j]

                # Ensure that the particles stay within the specified parameter ranges
                particles[j] = np.clip(particles[j], self.param_ranges[:, 0], self.param_ranges[:, 1])

        return global_best_particle_position, global_best_particle_score, logs


    def _generate_particle(self):
        return np.array([random.uniform(param_range[0], param_range[1]) for param_range in self.param_ranges])


    def plot_logs(self.):
        """Plot logs and save the figure"""
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(self.logs).T)
        plt.xlabel("Iteration")
        plt.title("Val accuracy")
        plt.savefig("logs.png")
        # plt.show()
