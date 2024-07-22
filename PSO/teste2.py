import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# Definindo a função Shaffer
def shaffer_function(position):
    x, y = position
    return 0.5 + (np.sin(np.sqrt(x**2 + y**2))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

# Classe Particle (conforme definido anteriormente)
class Particle:
    def __init__(self, x0, w=0.5, c1=1, c2=2):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pos_i = []
        self.vel_i = []

        self.num_dimensions = len(x0)

        self.best_pos_i = []
        self.err_i = float('inf')
        self.best_err_i = float('inf')

        for i in range(self.num_dimensions):
            self.pos_i.append(x0[i])
            self.vel_i.append(random.uniform(-1, 1))

    def evaluate(self, cost_function):
        self.err_i = cost_function(self.pos_i)

        if self.err_i < self.best_err_i:
            self.best_pos_i = list(self.pos_i)
            self.best_err_i = self.err_i

    def update_velocity(self, pos_best_g):
        for i in range(self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive = self.c1 * r1 * (self.best_pos_i[i] - self.pos_i[i])
            social = self.c2 * r2 * (pos_best_g[i] - self.pos_i[i])
            self.vel_i[i] = self.w * self.vel_i[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(self.num_dimensions):
            self.pos_i[i] = self.pos_i[i] + self.vel_i[i]

            if self.pos_i[i] > bounds[i][1]:
                self.pos_i[i] = bounds[i][1]
            if self.pos_i[i] < bounds[i][0]:
                self.pos_i[i] = bounds[i][0]

def pso(num_particles, num_iterations, bounds, cost_function):
    particles = []
    for _ in range(num_particles):
        initial_position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        particles.append(Particle(initial_position))

    best_pos_g = []
    best_err_g = float('inf')

    particle_positions = []
    fitness_values = []

    for t in range(num_iterations):
        current_positions = []
        current_fitness = []
        for particle in particles:
            particle.evaluate(cost_function)
            if particle.err_i < best_err_g:
                best_pos_g = list(particle.pos_i)
                best_err_g = particle.err_i

            current_positions.append(list(particle.pos_i))
            current_fitness.append(particle.err_i)
        
        particle_positions.append(current_positions)
        fitness_values.append(current_fitness)

        for particle in particles:
            particle.update_velocity(best_pos_g)
            particle.update_position(bounds)

    return best_pos_g, best_err_g, particle_positions, fitness_values

# Parâmetros
num_particles = 50
num_iterations = 100
bounds = [(-10, 10), (-10, 10)]

# Rodar o PSO
best_position, best_error, particle_positions, fitness_values = pso(num_particles, num_iterations, bounds, shaffer_function)

# Preparar dados para o gráfico
X = np.linspace(bounds[0][0], bounds[0][1], 100)
Y = np.linspace(bounds[1][0], bounds[1][1], 100)
X, Y = np.meshgrid(X, Y)
Z = 0.5 + (np.sin(np.sqrt(X**2 + Y**2))**2 - 0.5) / (1 + 0.001 * (X**2 + Y**2))**2

# Função para plotar as partículas em um gráfico 3D e 2D
def plot_particles(iteration, ax1, ax2):
    ax1.clear()
    ax2.clear()

    # Subplot 3D
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    current_positions = np.array(particle_positions[iteration])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], color='r', s=50)
    ax1.set_title(f'Iteração {iteration + 1} - Função Shaffer em 3D')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Subplot 2D
    contour = ax2.contourf(X, Y, Z, cmap='viridis')
    ax2.scatter(current_positions[:, 0], current_positions[:, 1], color='r', s=50)
    ax2.set_title(f'Iteração {iteration + 1} - Função Shaffer em 2D')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(contour, ax=ax2)

# Criar os gráficos para a 1ª, 50ª e última iteração
fig = plt.figure(figsize=(14, 7))

# 1ª iteração
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
plot_particles(0, ax1, ax2)
plt.show()

# Última iteração
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
plot_particles(num_iterations - 1, ax1, ax2)
plt.show()

# Gráfico de aptidão ao longo das iterações
best_fitness = [min(fitness) for fitness in fitness_values]
mean_fitness = [np.mean(fitness) for fitness in fitness_values]
worst_fitness = [max(fitness) for fitness in fitness_values]

plt.figure(figsize=(10, 6))
plt.plot(best_fitness, label='Melhor Aptidão')
plt.plot(mean_fitness, label='Aptidão Média')
plt.plot(worst_fitness, label='Pior Aptidão')
plt.title('Aptidão ao Longo das Iterações')
plt.xlabel('Iteração')
plt.ylabel('Aptidão')
plt.legend()
plt.show()

print(f"Melhor posição: {best_position}")
print(f"Melhor erro: {best_error}")
