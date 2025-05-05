import sys
import numpy as np
from collections import defaultdict

class RLAgent:
    def __init__(self, N, alpha=0.5, q_rate=0.1, jugador_agente=1):
        self.N = N                                      #training count
        self.alpha = alpha                              #learning rate
        self.q_rate = q_rate                            #q rate para e-greedy
        self.jugador_agente = jugador_agente            #jugador agente
        self.lookup_table = defaultdict(lambda: 0.5)    #default value p=0.5
        self.tablero = np.zeros((3, 3), dtype=int)      #tablero actual
        self.last_tablero = np.zeros((3, 3), dtype=int) #tablero anterior
        self.entrenar = True                            #entrenar o no
        self.game_result = 0                            #resultado del juego

    def reset(self, entrenar=True):
        """
        Reset del tablero, entrenamiento y resultado anterior
        """
        self.tablero = np.zeros((3, 3), dtype=int)
        self.last_tablero = np.zeros((3, 3), dtype=int)
        self.entrenar = entrenar
        self.game_result = 0

    def calculate_result(self, tablero):
        """
            Retorna:
                1: si ganó jugador 1
                2: si ganó jugador 2
                3: si hay empate
                0: si aún no hay resultado 
        """
        for jugador in [1, 2]:
            if any(np.all(tablero[row, :] == jugador) for row in range(3)) or \
               any(np.all(tablero[:, col] == jugador) for col in range(3)) or \
               np.all(np.diag(tablero) == jugador) or \
               np.all(np.diag(np.fliplr(tablero)) == jugador):
                return jugador
        return 3 if not np.any(tablero == 0) else 0

    def calculate_reward(self, tablero, jugador):
        contrario = 3 - jugador #3-1=2, 3-2=1
        result = self.calculate_result(tablero)
        if result == jugador:
            return 1.0 #ganó el jugador, reward +
        elif result == contrario:
            return 0.0 #perdió el jugador, reward 0
        elif result == 3:
            return 0.0 #empató el jugador, reward 0
        return self.get_probability(tablero)

    def get_probability(self, tablero):
        serialized = self.serialize_tablero(tablero)
        return self.lookup_table[serialized]

    def serialize_tablero(self, tablero):
        """
            010
            120 => 010120021
            021
        """
        return ''.join(map(str, tablero.flatten()))

    def deserialize_tablero(self, serialized):
        """
                         010
            010120021 => 120 
                         021
        """
        return np.array(list(map(int, serialized))).reshape(3, 3)

    def update_alpha(self, current_game):
        """
        Actualiza el valor de alpha basándose en el número actual de juegos.
        La fórmula reduce gradualmente alpha desde 0.5 a 0.01 durante el entrenamiento.
        Parámetros:
            current_game: int - Número de juegos completados hasta ahora.
        """
        self.alpha = 0.5 - 0.49 * current_game / self.N

    def update_probability(self, tablero, next_state_prob, jugador):
        """
        Update de la probabilidad en base a Bellman Expectation Equation
        """        
        prob = self.calculate_reward(tablero, jugador)
        prob += self.alpha * (next_state_prob - prob)
        self.lookup_table[self.serialize_tablero(tablero)] = prob

    def jugar(self, jugador):
        """
        Selecciona la siguiente jugada del agente basado en Montecarlo control
        arg max(p)
        Si está entrenando, actualiza probabilidades
        """
        max_prob = float('-inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if self.tablero[i, j] == 0:
                    self.tablero[i, j] = jugador
                    prob = self.calculate_reward(self.tablero, jugador)
                    if prob > max_prob:
                        max_prob = prob
                        best_move = (i, j)
                    self.tablero[i, j] = 0
        if self.entrenar: #actualizar probabilidad si estamos en entrenamiento
            self.update_probability(self.last_tablero, max_prob, jugador)
        if best_move:
            self.tablero[best_move] = jugador
            self.last_tablero = self.tablero.copy()

    def jugar_random(self, jugador):
        """
        Aleatoriza una jugada entre las posiciones vacías del tablero
        Si el jugador es el agente y está entrenando, actualiza probabilidades
        """
        empty_cells = np.argwhere(self.tablero == 0)
        if empty_cells.size:
            move = empty_cells[np.random.choice(len(empty_cells))]
            self.tablero[tuple(move)] = jugador
            if jugador == self.jugador_agente:
                if self.entrenar:
                    prob = self.calculate_reward(self.tablero, jugador)
                    self.update_probability(self.last_tablero, prob, jugador)
                self.last_tablero = self.tablero.copy()

    def jugar_vs_random(self):
        """
        Implementa 1 juego del agente vs jugador random
        El agente utiliza la politica Epsilon-greedy (q_rate) para entrenar
        """
        jugador = self.jugador_agente
        contrario = 3 - jugador
        turno = 1
        for _ in range(9):
            if turno == jugador:
                if np.random.random() <= self.q_rate or not self.entrenar:
                    self.jugar(jugador)
                else:
                    self.jugar_random(jugador)
            else:
                self.jugar_random(contrario)
            self.game_result = self.calculate_result(self.tablero)
            if self.game_result > 0:
                #si ganó el contrario, el último turno no fue del agente,
                #requiere actualizar probabilidades si estamos entrenando
                if self.game_result != jugador and self.entrenar:
                    self.update_probability(self.last_tablero, self.calculate_reward(self.tablero, jugador), jugador)
                break
            turno = 3 - turno
        
    def print_tablero(self, tablero = None) -> str:
        """
        Imprime el tablero de manera visual en un texto.
        Los valores son:
            1: 'x'
            2: 'o'
            0: ' ' (vacío)
        """
        if tablero is None: tablero = self.tablero
        output = []
        output.append("-------")
        for fila in tablero:
            row = "|".join(["x" if celda == 1 else "o" if celda == 2 else " " for celda in fila])
            output.append(f"|{row}|")
            output.append("-------")
        return "\n".join(output)
            
    def print_table(self, filename_suffix = ""):
        """
        Imprime todos los estados almacenados en la lookup_table con sus probabilidades.
        Cada estado se muestra junto con el tablero correspondiente.
        """
        with open(f"RLAgent_table{filename_suffix}.txt", "w") as file:
            for key, prob in self.lookup_table.items():
                file.write(f"\nTablero: {key}, probabilidad: {prob:.2f}")
                tablero = self.deserialize_tablero(key)
                file.write(self.print_tablero(tablero))

# Ejemplo de uso simple
"""
if __name__ == "__main__":
    agente = RLAgent(N=10000)
    for _ in range(agente.N):
        agente.reset(True)
        agente.jugar_vs_random()
    print("Entrenamiento completado.")
    agente.print_table()
"""

# Main de entrenamiento y validación
if __name__ == "__main__":

    # Parámetros de configuración
    training_count = 10000  # Número de partidas para entrenamiento
    total_games_count = 100  # Número de partidas para validación
    total_experiments = 1  # Número de experimentos a realizar
    print_details = True # Imprimir detalles de entrenamiento y validación
    print_lookup_table = True  # Imprimir tabla de búsqueda después del entrenamiento
    #q_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Valores de q_rate
    q_rates = [0.3, 0.7]  # Valores de q_rate
    
    for q_rate in q_rates:
        wins_ratio_acum = 0
        losses_ratio_acum = 0
        draws_ratio_acum = 0
        
        sys.stdout.write("\n")
        for e_index in range(total_experiments):
            # Entrenamiento
            agente = RLAgent(N=training_count)
            agente.q_rate = q_rate
            for i in range(agente.N):
                if print_details:
                    sys.stdout.write(f"\r>>> QRate: {q_rate}, Experiment:{e_index+1}/{total_experiments}, Training: {i+1} from {agente.N}")
                agente.reset(entrenar=True)
                agente.update_alpha(i)
                agente.jugar_vs_random()
            sys.stdout.write("\n")

            # Imprimir tabla de búsqueda después del entrenamiento
            if print_lookup_table:
                sys.stdout.write(">>> Tabla de búsqueda después del entrenamiento:")
                agente.print_table(f"_{q_rate}")

            # Validación
            wins, losses, draws = 0, 0, 0
            contrario = 3 - agente.jugador_agente
            for i in range(int(total_games_count)):
                
                if print_details:
                    sys.stdout.write(f"\r>>> QRate: {q_rate}, Experiment:{e_index+1}/{total_experiments}, Validation: {i+1} from {total_games_count}")
                agente.reset(entrenar=False)
                agente.jugar_vs_random()

                if agente.game_result == agente.jugador_agente:
                    wins += 1
                elif agente.game_result == contrario:
                    losses += 1
                else:
                    draws += 1
            sys.stdout.write("\n")

            wins_ratio_acum += wins / total_games_count
            losses_ratio_acum += losses / total_games_count
            draws_ratio_acum += draws / total_games_count

        # Imprimir resultados promedio por tasa de Q
        sys.stdout.write(f">>>>>>>>>>>>> RATIO AVG, Q RATE: {q_rate}")
        sys.stdout.write(f"\nRatio Avg W/T: {wins_ratio_acum / total_experiments:.4f}")
        sys.stdout.write(f"\nRatio Avg L/T: {losses_ratio_acum / total_experiments:.4f}")
        sys.stdout.write(f"\nRatio Avg D/T: {draws_ratio_acum / total_experiments:.4f}")
        sys.stdout.write("\n")

