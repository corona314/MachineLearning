import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import time
import argparse
import warnings
import os
import statistics
from time import sleep
from random import randint
from math import ceil, floor
import matplotlib.pyplot as plt
import pandas as pd

# Intento de import qiskit/torch; el script funcionará con solo el agente clásico si faltan libs.
# ----------------------------
# Dependencias (imports robustos)
# ----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
try:
    # circuit libs
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    # paulis / observables
    from qiskit.quantum_info import SparsePauliOp, Pauli
    # AerSimulator (intento qiskit_aer primero, luego providers.aer)
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        from qiskit import AerSimulator
    # Estimator primitive (puede existir)
    try:
        from qiskit.primitives import Estimator as PrimitiveEstimator
    except Exception:
        PrimitiveEstimator = None
    # qiskit-ml
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit import transpile

    QISKIT_AVAILABLE = True
except Exception as e:
    QISKIT_AVAILABLE = False
    _qiskit_import_error = e

# Mensaje de diagnóstico
if not QISKIT_AVAILABLE:
    print("WARNING: Qiskit no cargado correctamente. Error:", _qiskit_import_error)
else:
    print("Qiskit cargado. AerSimulator disponible?", 'Yes' if 'AerSimulator' in globals() and AerSimulator is not None else 'No')


class PongEnvironment:
    
    def __init__(self, max_life=3, height_px = 40, width_px = 50, movimiento_px = 3, default_reward = 10):
        
        self.action_space = ['Arriba','Abajo']
        
        self.default_reward = default_reward
        
        self._step_penalization = 0
        
        self.state = [0,0,0]
        
        self.total_reward = 0
        
        self.dx = movimiento_px
        self.dy = movimiento_px
        
        filas = ceil(height_px/movimiento_px)
        columnas = ceil(width_px/movimiento_px)
        
        self.positions_space = np.array([[[0 for z in range(columnas)] for y in range(filas)] for x in range(filas)])

        self.lives = max_life
        self.max_life=max_life
        
        self.x = randint(int(width_px/2), width_px) 
        self.y = randint(0, height_px-10)
        
        self.player_alto = int(height_px/4)

        self.player1 = self.player_alto  # posic. inicial del player
        
        self.score = 0
        
        self.width_px = width_px
        self.height_px = height_px
        self.radio = 2.5

    def reset(self):
        self.total_reward = 0
        self.state = [0,0,0]
        self.lives = self.max_life
        self.score = 0
        self.x = randint(int(self.width_px/2), self.width_px) 
        self.y = randint(0, self.height_px-10)
        return self.state

    def step(self, action, animate=False, custom_reward=None):
        if custom_reward is None:
            custom_reward = self.default_reward
        self._apply_action(action, animate, custom_reward)
        done = self.lives <=0 # final
        reward = self.score
        reward += self._step_penalization
        self.total_reward += reward
        return self.state, reward , done

    def _apply_action(self, action, animate=False, custom_reward=10):
        
        if action == "Arriba":
            self.player1 += abs(self.dy)
        elif action == "Abajo":
            self.player1 -= abs(self.dy)

        self.avanza_player()

        self.avanza_frame(custom_reward)

        if animate:
            if not hasattr(self, '_fig_ax'):
                self._fig_ax = self.dibujar_frame()  # crea la figura y eje la primera vez
            else:
                self._fig_ax = self.dibujar_frame(self._fig_ax)  # refresca la misma figura

        self.state = (floor(self.player1/abs(self.dy))-2, floor(self.y/abs(self.dy))-2, floor(self.x/abs(self.dx))-2)
    
    def detectaColision(self, ball_y, player_y):
        if (player_y+self.player_alto >= (ball_y-self.radio)) and (player_y <= (ball_y+self.radio)):
            return True
        else:
            return False
    
    def avanza_player(self):
        if self.player1 + self.player_alto >= self.height_px:
            self.player1 = self.height_px - self.player_alto
        elif self.player1 <= -abs(self.dy):
            self.player1 = -abs(self.dy)

    def avanza_frame(self, reward):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 3 or self.x > self.width_px:
            self.dx = -self.dx
            if self.x <= 3:
                ret = self.detectaColision(self.y, self.player1)

                if ret:
                    self.score = reward # recompensa positiva
                else:
                    self.score = -reward # recompensa negativa
                    self.lives -= 1
                    if self.lives>0:
                        self.x = randint(int(self.width_px/2), self.width_px)
                        self.y = randint(0, self.height_px-10)
                        self.dx = abs(self.dx)
                        self.dy = abs(self.dy)
        else:
            self.score = 0

        if self.y < 0 or self.y > self.height_px:
            self.dy = -self.dy

    def dibujar_frame(self, fig_ax=None):
        # Si no existe la figura, crearla
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(5,4))
        else:
            fig, ax = fig_ax
            ax.clear()  # limpiar el frame anterior

        # dibujar la bola
        circle = plt.Circle((self.x, self.y), self.radio, fc='slategray', ec="black")
        ax.add_patch(circle)
        # dibujar el jugador
        rectangle = plt.Rectangle((-5, self.player1), 5, self.player_alto, fc='gold', ec="none")
        ax.add_patch(rectangle)

        # límites y texto
        ax.set_xlim(-5, self.width_px+5)
        ax.set_ylim(-5, self.height_px+5)
        ax.text(4, self.height_px, f"SCORE:{self.total_reward}  LIFE:{self.lives}", fontsize=12)
        if self.lives <=0:
            ax.text(10, self.height_px-14, "GAME OVER", fontsize=16)
        elif self.total_reward >= 1000:
            ax.text(10, self.height_px-14, "YOU WIN!", fontsize=16)

        plt.pause(0.001)  # pausa corta para que se refresque la ventana
        sleep(0.02)
        return fig, ax

class QuantumPongAgent:

    def __init__(
        self,
        game,
        discount_factor=0.2,
        learning_rate=0.01,
        ratio_explotacion=0.85,
        device="cpu"
    ):

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        self.device = device

        # Estados: [player_pos, ball_y, ball_x] → 3 features → 3 qubits
        self.num_qubits = 3
        self.num_actions = len(game.action_space)  # 2

        # --- Construcción del QNN ---
        feature_map = ZZFeatureMap(self.num_qubits, reps=1)
        ansatz = RealAmplitudes(self.num_qubits, reps=1)

        qc = feature_map.compose(ansatz)

        # Creamos 2 observables (uno por acción)
        observables = [
            SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1.0)]),
            SparsePauliOp.from_list([("I" + "Z" + "I" * (self.num_qubits - 2), 1.0)])
        ]

        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.model = TorchConnector(qnn).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    # -----------------------------
    # Transformar estado → tensor
    # -----------------------------
    def _state_to_tensor(self, state):
        # Normalización ligera igual que tu versión anterior
        s = torch.tensor([
            state[0] / 50,
            state[1] / 50,
            state[2] / 50
        ], dtype=torch.float32, device=self.device)
        return s

    # -----------------------------
    # Política ε-greedy
    # -----------------------------
    def get_next_step(self, state, game):

        # acción aleatoria
        if np.random.rand() > self.ratio_explotacion:
            return np.random.choice(list(game.action_space))

        # acción greedy sobre QNN
        with torch.no_grad():
            s = self._state_to_tensor(state)
            qvals = self.model(s)            # vector tamaño 2
            action_idx = torch.argmax(qvals).item()

        return list(game.action_space)[action_idx]

    # -----------------------------
    # Actualización Q-learning
    # -----------------------------
    def update(self, game, old_state, action_taken, reward, new_state, reached_end):

        # Índice de acción
        idx = list(game.action_space).index(action_taken)

        s_old = self._state_to_tensor(old_state)
        s_new = self._state_to_tensor(new_state)

        # Q(s,a)
        q_old = self.model(s_old)

        with torch.no_grad():
            q_next = self.model(s_new)
            max_future = torch.max(q_next).item()

        # Target Q
        if reached_end:
            target_value = reward
        else:
            target_value = reward + self.discount_factor * max_future

        target_tensor = torch.tensor([target_value], dtype=torch.float32, device=self.device)

        # Loss = (Q(s,a) - target)^2
        prediction = q_old[idx:idx+1]

        loss = self.loss_fn(prediction, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    # Para depuración
    def print_q_example(self, sample_state=(0,0,0)):
        print("Q(sample_state) =", self.model(self._state_to_tensor(sample_state)))

def play_quantum(
    rounds=5000,
    max_life=3,
    discount_factor=0.1,
    learning_rate=0.01,
    ratio_explotacion=0.85,
    learner=None,
    game=None,
    animate=False,
    device="cpu"
):

    if game is None:
        game = PongEnvironment(max_life=max_life, movimiento_px=3)

    if learner is None:
        print("Begin new Quantum Train!")
        learner = QuantumPongAgent(
            game,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            ratio_explotacion=ratio_explotacion,
            device=device
        )

    max_points = -9999
    first_max_reached = 0
    total_rw = 0
    steps = []

    for played in range(rounds):
        state = game.reset()
        reward, done = None, None
        it = 0

        while (not done) and (it < 3000 and game.total_reward <= 1000):
            old_state = np.array(state)
            action = learner.get_next_step(state, game)
            state, reward, done = game.step(action, animate=animate)

            if rounds > 1:
                learner.update(game, old_state, action, reward, state, done)

            it += 1

        steps.append(it)
        total_rw += game.total_reward

        if game.total_reward > max_points:
            max_points = game.total_reward
            first_max_reached = played

        if played % 10 == 0 and played > 1 and not animate:
            print(
                f"-- Games[{played}] Avg[{int(total_rw/played)}] "
                f"Steps[{int(np.mean(steps))}] Max[{max_points}]"
            )
    if played>1:
        print(f"Games[{played}] Avg[{int(total_rw/played)}] Max[{max_points}] first at[{first_max_reached}]")

    return learner, game

learner_q, game_q = play_quantum(
    rounds=20,
    discount_factor=0.2,
    learning_rate=0.01,
    ratio_explotacion=0.85,
    device="cpu"
)

# Modo greedy para ver comportamiento entrenado
learner_demo = learner_q
learner_demo.ratio_explotacion = 1.0

demo_game = PongEnvironment(max_life=3, movimiento_px=3)
player = play_quantum(rounds=1, learner=learner_demo, game=demo_game, animate=True)