from particle import *
from function import *

def pso(num_particles, num_iterations, bounds, cost_function):
    # Inicializa uma lista de partículas
    particles = []
    for _ in range(num_particles):
        initial_position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        particles.append(Particle(initial_position))

    # Melhor posição global
    best_pos_g = []
    best_err_g = float('inf')

    # Itera sobre o número de iterações
    for t in range(num_iterations):
        for particle in particles:
            # Avalia o fitness atual da partícula
            particle.evaluate(cost_function)

            # Verifica se a posição atual é a melhor global
            if particle.err_i < best_err_g:
                best_pos_g = list(particle.pos_i)
                best_err_g = particle.err_i

        for particle in particles:
            # Atualiza a velocidade e a posição de cada partícula
            particle.update_velocity(best_pos_g)
            particle.update_position(bounds)

    return best_pos_g, best_err_g

# Parâmetros
num_particles = 30
num_iterations = 100
bounds = [(-10, 10), (-10, 10), (-10, 10)]  # Limites para cada dimensão

# Rodar o PSO
best_position, best_error = pso(num_particles, num_iterations, bounds, sphere_function)

print(f"Melhor posição: {best_position}")
print(f"Melhor erro: {best_error}")
