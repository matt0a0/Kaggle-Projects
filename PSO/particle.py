import random

class Particle:
    def __init__(self, x0, w=0.5, c1=1, c2=2):
        # Propriedades de uma única partícula
        self.w = w  # Peso inercial
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social
        self.pos_i = []  # Posição da partícula
        self.vel_i = []  # Velocidade da partícula

        self.num_dimensions = len(x0)  # Número de dimensões

        self.best_pos_i = []  # Melhor posição individual
        self.err_i = float('inf')  # Erro atual
        self.best_err_i = float('inf')  # Melhor erro individual

        # Inicializa a posição e a velocidade da partícula
        for i in range(0,self.num_dimensions):
            self.pos_i.append(x0[i])
            self.vel_i.append(random.uniform(-1, 1))

    def evaluate(self, cost_function):
        # Avalia o fitness atual
        self.err_i = cost_function(self.pos_i)

        # Verifica se a posição atual é a melhor individual
        if self.err_i < self.best_err_i:
            self.best_pos_i = list(self.pos_i)
            self.best_err_i = self.err_i

    def update_velocity(self, pos_best_g):
        # Atualiza a velocidade da partícula
        for i in range(self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive = self.c1 * r1 * (self.best_pos_i[i] - self.pos_i[i])
            social = self.c2 * r2 * (pos_best_g[i] - self.pos_i[i])
            self.vel_i[i] = self.w * self.vel_i[i] + cognitive + social

    def update_position(self, bounds):
        # Atualiza a posição da partícula baseada na nova velocidade
        for i in range(self.num_dimensions):
            self.pos_i[i] = self.pos_i[i] + self.vel_i[i]

            # Verifica os limites superiores e inferiores
            if self.pos_i[i] > bounds[i][1]:
                self.pos_i[i] = bounds[i][1]
            if self.pos_i[i] < bounds[i][0]:
                self.pos_i[i] = bounds[i][0]
