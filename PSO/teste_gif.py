import os
import glob
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random

# Definindo a função da esfera
def sphere_function(position):
    return sum(x**2 for x in position)

# Classe Particle (conforme definido anteriormente)
class Particle:
    def __init__(self, x0,w=0.72892, c1=2.05,c2=2.05):
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

def pso(num_particles, num_iterations, bounds, cost_function, w=0.5, c1=1, c2=2):
    particles = []
    for _ in range(num_particles):
        initial_position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        particles.append(Particle(initial_position, w, c1, c2))

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

# Função para executar múltiplas execuções do PSO
def multiple_pso_executions(num_executions, num_particles, num_iterations, bounds, cost_function, w=0.5, c1=1, c2=2):
    best_positions = []
    best_errors = []
    all_particle_positions = []
    all_fitness_values = []

    for i in range(num_executions):
        best_pos, best_err, particle_positions, fitness_values = pso(num_particles, num_iterations, bounds, cost_function, w, c1, c2)
        best_positions.append(best_pos)
        best_errors.append(best_err)
        all_particle_positions.append(particle_positions)
        all_fitness_values.append(fitness_values)
        print(f'Execução {i+1}: Melhor Posição = {best_pos}, Melhor Erro = {best_err}')

    return best_positions, best_errors, all_particle_positions, all_fitness_values

# Função para plotar partículas
def plot_particles(iteration, particle_positions, ax1, ax2, bounds, best_pos, best_err, exec_num, is_final, cost_function):
    X = np.linspace(bounds[0][0], bounds[0][1], 100)
    Y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([cost_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    ax1.clear()
    ax2.clear()

    # Subplot 3D
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    current_positions = np.array(particle_positions[iteration])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], color='r', s=50)
    ax1.set_title(f'Execução {exec_num+1} - Iteração {iteration + 1} - Função em 3D')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Subplot 2D
    contour = ax2.contourf(X, Y, Z, cmap='viridis')
    ax2.scatter(current_positions[:, 0], current_positions[:, 1], color='r', s=50)
    ax2.set_title(f'Execução {exec_num+1} - Iteração {iteration + 1} - Função em 2D')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(contour, ax=ax2)

    if is_final:
        ax2.text(0.05, 0.95, f'Melhor Fitness: {best_err:.4f}\nPosição: {best_pos}', transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Função para criar GIF a partir de uma pasta
def make_gif_from_folder(folder, out_file_path, remove_folder=True):
    files = os.path.join(folder, '*.png')
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(files))]
    img.save(fp=out_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    if remove_folder:
        shutil.rmtree(folder, ignore_errors=True)

# Criar diretório para salvar as imagens
output_dir = 'pso_executions_sphere'
os.makedirs(output_dir, exist_ok=True)

# Parâmetros
num_particles = 100
num_iterations = 100
bounds = [(-10, 10), (-10, 10)]
num_executions = 10

# Executar múltiplas execuções do PSO
best_positions, best_errors, all_particle_positions, all_fitness_values = multiple_pso_executions(num_executions, num_particles, num_iterations, bounds, sphere_function)

average_best_fitness = np.mean(best_errors)
average_best_positions = np.mean(best_positions, axis=0)

# Gráfico de aptidão ao longo das iterações (média das 10 execuções)
best_fitness_per_iter = np.mean([[min(fitness) for fitness in execution] for execution in all_fitness_values], axis=0)
mean_fitness_per_iter = np.mean([[np.mean(fitness) for fitness in execution] for execution in all_fitness_values], axis=0)
worst_fitness_per_iter = np.mean([[max(fitness) for fitness in execution] for execution in all_fitness_values], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(best_fitness_per_iter, label='Melhor Aptidão')
plt.plot(mean_fitness_per_iter, label='Aptidão Média')
plt.plot(worst_fitness_per_iter, label='Pior Aptidão')
plt.title('Aptidão ao Longo das Iterações (Média de 10 Execuções)')
plt.xlabel('Iteração')
plt.ylabel('Aptidão')
plt.legend()
plt.savefig(os.path.join(output_dir, 'fitness_over_iterations.png'))
plt.close()

# Definir a execução específica para gerar o GIF (nesse caso, a primeira execução)
exec_num = 0

# Criar diretório temporário para salvar as imagens
tmp_dir = os.path.join(output_dir, f'_tmp_exec_{exec_num+1}')
os.makedirs(tmp_dir, exist_ok=True)

# Gerar e salvar imagens para cada iteração da execução específica
for iteration in range(num_iterations):
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    plot_particles(iteration, all_particle_positions[exec_num], ax1, ax2, bounds, best_positions[exec_num], best_errors[exec_num], exec_num, iteration == num_iterations - 1, sphere_function)
    plt.savefig(os.path.join(tmp_dir, f'{iteration:05d}.png'))
    plt.close()

# Gerar o GIF a partir das imagens salvas
make_gif_from_folder(tmp_dir, os.path.join(output_dir, f'exec_{exec_num+1}.gif'))

print(f"Média da melhor posição: {average_best_positions}")
print(f"Média do melhor erro: {average_best_fitness}")