#!/usr/bin/env python3
# experiment_multicore.py
"""
Versión multinúcleo de experiment.py — ejecuta seeds/agents en paralelo usando ProcessPoolExecutor.
Mantiene compatibilidad con el resto del código original.
"""

import time
import argparse
import warnings
import os
import statistics
import numpy as np
from time import sleep
from random import randint
from math import ceil, floor
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import traceback

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
    print("WARNING: Qiskit no cargado correctamente. Error:", getattr(globals(), '_qiskit_import_error', 'unknown'))
else:
    print("Qiskit cargado. AerSimulator disponible?", 'Yes' if 'AerSimulator' in globals() and AerSimulator is not None else 'No')


# ---------------------------
# Entorno
# ---------------------------
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
        self.player1 = self.player_alto
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
        done = self.lives <=0
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
                self._fig_ax = self.dibujar_frame()
            else:
                self._fig_ax = self.dibujar_frame(self._fig_ax)
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
                    self.score = reward
                else:
                    self.score = -reward
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
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(5,4))
        else:
            fig, ax = fig_ax
            ax.clear()
        circle = plt.Circle((self.x, self.y), self.radio, fc='slategray', ec="black")
        ax.add_patch(circle)
        rectangle = plt.Rectangle((-5, self.player1), 5, self.player_alto, fc='gold', ec="none")
        ax.add_patch(rectangle)
        ax.set_xlim(-5, self.width_px+5)
        ax.set_ylim(-5, self.height_px+5)
        ax.text(4, self.height_px, f"SCORE:{self.total_reward}  LIFE:{self.lives}", fontsize=12)
        if self.lives <=0:
            ax.text(10, self.height_px-14, "GAME OVER", fontsize=16)
        elif self.total_reward >= 1000:
            ax.text(10, self.height_px-14, "YOU WIN!", fontsize=16)
        plt.pause(0.001)
        sleep(0.02)
        return fig, ax

# ---------------------------
# Agente clásico
# ---------------------------
class PongAgent:
    def __init__(self, game, policy=None, discount_factor = 0.1, learning_rate = 0.1, ratio_explotacion = 0.9):
        if policy is not None:
            self._q_table = policy
        else:
            position = list(game.positions_space.shape)
            position.append(len(game.action_space))
            self._q_table = np.zeros(position)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        self.forward_calls = 0
        self.backward_calls = 0

    def get_next_step(self, state, game):
        self.forward_calls += 1
        next_step = np.random.choice(list(game.action_space))
        if np.random.uniform() <= self.ratio_explotacion:
            idx_action = np.random.choice(np.flatnonzero(
                    self._q_table[state[0],state[1],state[2]] == self._q_table[state[0],state[1],state[2]].max()
                ))
            next_step = list(game.action_space)[idx_action]
        return next_step

    def update(self, game, old_state, action_taken, reward_action_taken, new_state, reached_end):
        self.backward_calls += 1
        idx_action_taken =list(game.action_space).index(action_taken)
        actual_q_value_options = self._q_table[old_state[0], old_state[1], old_state[2]]
        actual_q_value = actual_q_value_options[idx_action_taken]
        future_q_value_options = self._q_table[new_state[0], new_state[1], new_state[2]]
        future_max_q_value = reward_action_taken  +  self.discount_factor*future_q_value_options.max()
        if reached_end:
            future_max_q_value = reward_action_taken
        self._q_table[old_state[0], old_state[1], old_state[2], idx_action_taken] = actual_q_value + \
                                              self.learning_rate*(future_max_q_value -actual_q_value)

    def get_policy(self):
        return self._q_table

# ---------------------------
# Agente cuántico
# ---------------------------
class QuantumPongAgent:
    def __init__(self, game, discount_factor=0.1, learning_rate=1e-3,
                 ratio_explotacion=0.9, num_qubits=3, reps=0, device='cpu'):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit/Torch no están disponibles en este entorno.")
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        self.action_space = game.action_space
        self.num_actions = len(game.action_space)
        self.num_qubits = num_qubits
        self.device = torch.device(device)
        self.forward_calls = 0
        self.backward_calls = 0
        self.feature_map = ZZFeatureMap(self.num_qubits)
        self.ansatz = RealAmplitudes(self.num_qubits, reps=reps)
        self.qc = self.feature_map.compose(self.ansatz)

        try:
            obs = []
            for i in range(self.num_actions):
                qubit_idx = i % self.num_qubits
                pauli_str = ['I'] * self.num_qubits
                pauli_str[qubit_idx] = 'Z'
                pauli_label = ''.join(reversed(pauli_str))
                obs.append(SparsePauliOp(Pauli(pauli_label)))

            try:
                backend = AerSimulator(method='statevector')
                backend.set_options(max_parallel_threads=int(os.environ.get("OMP_NUM_THREADS","4")))
                primitive_estimator = PrimitiveEstimator(options={'backend': backend})
            except Exception:
                primitive_estimator = None

            if primitive_estimator is not None:
                self.qnn = EstimatorQNN(circuit=self.qc,
                                        observables=obs,
                                        input_params=self.feature_map.parameters,
                                        weight_params=self.ansatz.parameters,
                                        estimator=primitive_estimator)
            else:
                self.qnn = EstimatorQNN(circuit=self.qc,
                                        observables=obs,
                                        input_params=self.feature_map.parameters,
                                        weight_params=self.ansatz.parameters)
            self.model = TorchConnector(self.qnn).to(self.device)
            self.model.eval()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.MSELoss()
            self.mode = "estimator_multi"
            try:
                if primitive_estimator is not None:
                    transpile(self.qc, backend)
            except Exception:
                pass

        except Exception as e:
            warnings.warn("Fallo creando EstimatorQNN multi-output: usar fallback statevector_manual. Error: " + str(e))
            self.mode = "statevector_manual"
            self._setup_statevector_manual()

    def _setup_statevector_manual(self):
        self.backend = AerSimulator(method='statevector')
        self.backend.set_options(max_parallel_threads=int(os.environ.get("OMP_NUM_THREADS","4")))
        try:
            self.transpiled = transpile(self.qc, self.backend)
        except Exception:
            self.transpiled = self.qc
        self.input_params = list(self.feature_map.parameters)
        self.weight_params = list(self.ansatz.parameters)

        class SimpleAnsatzModule(nn.Module):
            def __init__(self, n_weights):
                super().__init__()
                self.weights = nn.Parameter(0.01 * torch.randn(n_weights))
            def forward(self):
                return self.weights

        n_weights = len(self.weight_params)
        self.ansatz_module = SimpleAnsatzModule(n_weights).to(self.device)
        self.optimizer = optim.Adam(self.ansatz_module.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        from qiskit.quantum_info import SparsePauliOp
        self.obs_ops = []
        for i in range(self.num_actions):
            qubit_idx = i % self.num_qubits
            pauli_str = ['I'] * self.num_qubits
            pauli_str[qubit_idx] = 'Z'
            pauli_label = ''.join(reversed(pauli_str))
            self.obs_ops.append(SparsePauliOp(Pauli(pauli_label)))

    def state_to_tensor(self, state):
        return torch.tensor([s / 50 for s in state], dtype=torch.float32, device=self.device)

    @staticmethod
    def _expectation_from_statevector(statevector, observable: SparsePauliOp):
        mat = observable.to_matrix()
        psi = statevector
        exp = np.vdot(psi, mat.dot(psi))
        return np.real_if_close(exp).item()

    def _run_statevector_and_get_outputs(self, param_values, input_values):
        bind_dict = {}
        for i, p in enumerate(self.input_params):
            bind_dict[p] = float(input_values[i % len(input_values)])
        for i, p in enumerate(self.weight_params):
            bind_dict[p] = float(param_values[i])
        bound = self.transpiled.assign_parameters(bind_dict)
        qobj = self.backend.run(bound)
        res = qobj.result()
        sv = res.get_statevector(bound)
        outputs = []
        for obs in self.obs_ops:
            outputs.append(float(self._expectation_from_statevector(np.array(sv), obs)))
        return outputs

    def _forward_outputs(self, state_tensor):
        self.forward_calls += 1
        if self.mode == "estimator_multi":
            with torch.no_grad():
                out = self.model(state_tensor)
                try:
                    vals = out.detach().cpu().numpy().reshape(-1).tolist()
                except Exception:
                    vals = [float(out.item())] if out.numel()==1 else out.cpu().numpy().reshape(-1).tolist()
            return vals
        else:
            param_values = self.ansatz_module.weights.detach().cpu().numpy()
            input_values = state_tensor.detach().cpu().numpy().astype(float)
            outputs = self._run_statevector_and_get_outputs(param_values, input_values)
            return outputs

    def get_next_step(self, state):
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_vals = self._forward_outputs(state_tensor)
        q_tensor = np.array(q_vals)
        if np.random.uniform() <= self.ratio_explotacion:
            return self.action_space[int(np.argmax(q_tensor))]
        else:
            return np.random.choice(self.action_space)

    def update(self, state, action_taken, reward, next_state, done):
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        with torch.no_grad():
            q_vals = np.array(self._forward_outputs(state_tensor))
            q_next = np.array(self._forward_outputs(next_state_tensor))
        target = reward
        if not done:
            target += self.discount_factor * float(q_next.max())
        action_index = self.action_space.index(action_taken)
        self.backward_calls += 1
        if self.mode == "estimator_multi":
            self.optimizer.zero_grad()
            self.model.train()
            output = self.model(state_tensor)
            target_vec = output.detach().clone()
            target_vec[action_index] = torch.tensor(target, dtype=target_vec.dtype, device=target_vec.device)
            loss = self.loss_fn(output, target_vec)
            loss.backward()
            self.optimizer.step()
            self.model.eval()
        else:
            eps = 1e-3
            w = self.ansatz_module.weights.detach().clone()
            grad = torch.zeros_like(w)
            base_outputs = np.array(self._run_statevector_and_get_outputs(w.cpu().numpy(), state_tensor.cpu().numpy()))
            base_val = base_outputs[action_index]
            for i in range(len(w)):
                w_plus = w.clone()
                w_minus = w.clone()
                w_plus[i] += eps
                w_minus[i] -= eps
                out_p = np.array(self._run_statevector_and_get_outputs(w_plus.cpu().numpy(), state_tensor.cpu().numpy()))
                out_m = np.array(self._run_statevector_and_get_outputs(w_minus.cpu().numpy(), state_tensor.cpu().numpy()))
                grad_i = (out_p[action_index] - out_m[action_index]) / (2*eps)
                grad[i] = float(grad_i)
            pred = base_val
            g_loss = 2.0 * (pred - target) * grad
            with torch.no_grad():
                self.ansatz_module.weights -= (self.learning_rate * g_loss)


# ---------------------------
# Runner de experimentos (idéntico, serial)
# ---------------------------
def run_single(agent_type, mode, budget, seed, agent_kwargs, game_kwargs):
    np.random.seed(seed)
    game = PongEnvironment(**game_kwargs)
    if agent_type == 'classical':
        agent = PongAgent(game, **agent_kwargs)
    elif agent_type == 'quantum':
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit/Torch no disponibles para agente cuántico.")
        agent = QuantumPongAgent(game, **agent_kwargs)
    else:
        raise ValueError("agent_type must be 'classical' or 'quantum'")

    ep_rewards = []
    ep_times = []
    ep_forward_calls = []
    ep_backward_calls = []

    start_total = time.time()
    episodes_done = 0

    while True:
        if mode == 'time' and (time.time() - start_total) >= budget:
            break
        if mode == 'episodes' and episodes_done >= budget:
            break

        s = game.reset()
        done = False
        ep_start = time.time()
        steps = 0
        while not done and steps < 3000 and game.total_reward <= 1000:
            if agent_type == 'classical':
                a = agent.get_next_step(s, game)
            else:
                a = agent.get_next_step(s)
            next_s, r, done = game.step(a)
            if agent_type == 'classical':
                agent.update(game, s, a, r, next_s, done)
            else:
                agent.update(s, a, r, next_s, done)
            s = next_s
            steps += 1
        ep_time = time.time() - ep_start
        ep_rewards.append(game.total_reward)
        ep_times.append(ep_time)
        ep_forward_calls.append(getattr(agent, 'forward_calls', 0))
        ep_backward_calls.append(getattr(agent, 'backward_calls', 0))
        episodes_done += 1

    total_time = time.time() - start_total
    summary = {
        'agent_type': agent_type,
        'seed': seed,
        'episodes_done': episodes_done,
        'total_time_s': total_time,
        'avg_reward_per_ep': float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        'std_reward_per_ep': float(np.std(ep_rewards)) if ep_rewards else 0.0,
        'avg_time_per_ep': float(np.mean(ep_times)) if ep_times else 0.0,
        'total_forward_calls': int(ep_forward_calls[-1]) if ep_forward_calls else 0,
        'total_backward_calls': int(ep_backward_calls[-1]) if ep_backward_calls else 0,
        'rewards': ep_rewards,
        'times': ep_times
    }
    return summary

def aggregate_and_print(results, name):
    seeds = [r['seed'] for r in results]
    episodes = [r['episodes_done'] for r in results]
    times = [r['total_time_s'] for r in results]
    avg_rewards = [r['avg_reward_per_ep'] for r in results]
    print(f"\n=== RESUMEN {name} ===")
    print(f"Repeticiones: {len(results)}, Seeds: {seeds}")
    print(f"Episodes per run: mean={statistics.mean(episodes):.1f} std={statistics.pstdev(episodes):.1f}")
    print(f"Tiempo total (s): mean={statistics.mean(times):.1f} std={statistics.pstdev(times):.1f}")
    print(f"Avg reward/ep: mean={statistics.mean(avg_rewards):.3f} std={statistics.pstdev(avg_rewards):.3f}")
    return pd.DataFrame([{
        'seed': r['seed'],
        'episodes_done': r['episodes_done'],
        'total_time_s': r['total_time_s'],
        'avg_reward_per_ep': r['avg_reward_per_ep'],
        'std_reward_per_ep': r['std_reward_per_ep'],
        'avg_time_per_ep': r['avg_time_per_ep'],
        'total_forward_calls': r['total_forward_calls'],
        'total_backward_calls': r['total_backward_calls']
    } for r in results])

def save_detail(results, fname):
    rows = []
    for r in results:
        for i, rew in enumerate(r['rewards']):
            rows.append({'seed': r['seed'], 'ep': i, 'reward': rew})
    pd.DataFrame(rows).to_csv(fname, index=False)


def run_tasks_in_parallel(tasks, mode, budget, agent_kwargs, game_kwargs, workers):
    """
    tasks: list of tuples (agent_type, seed)
    devuelve: dict with lists: classical_results, quantum_results
    """
    classical_results = []
    quantum_results = []
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for agent_type, seed in tasks:
            fut = ex.submit(run_single, agent_type, mode, budget, seed, agent_kwargs, game_kwargs)
            futures[fut] = (agent_type, seed)
        for fut in concurrent.futures.as_completed(futures):
            agent_type, seed = futures[fut]
            try:
                res = fut.result()
                if res['agent_type'] == 'classical':
                    classical_results.append(res)
                else:
                    quantum_results.append(res)
                print(f"Task finished: {agent_type} seed={seed} episodes={res['episodes_done']} time={res['total_time_s']:.1f}s avg_r={res['avg_reward_per_ep']:.2f}")
            except Exception as e:
                print(f"Task failed: {agent_type} seed={seed}. Error:")
                traceback.print_exc()
    # ordenar por seed
    classical_results.sort(key=lambda x: x['seed'])
    quantum_results.sort(key=lambda x: x['seed'])
    return classical_results, quantum_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['time','episodes'], default='time', help='Modo de igualación')
    parser.add_argument('--budget', type=float, default=600.0, help='Segundos (time) o episodios (episodes)')
    parser.add_argument('--repeats', type=int, default=1, help='Repeticiones por agente (seeds)')
    parser.add_argument('--out', default='results', help='Directorio de salida para CSV/plots')
    parser.add_argument('--agents', default='classical,quantum', help='Qué agentes correr: classical, quantum, o ambos separados por coma')
    parser.add_argument('--reward', type=float, default=10.0, help='Recompensa por golpe de pelota (default=10)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1, help='Número de procesos en paralelo')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    seeds = list(range(args.repeats))
    selected_agents = [a.strip() for a in args.agents.split(',')]
    tasks = []
    for s in seeds:
        for a in selected_agents:
            if a == 'quantum' and not QISKIT_AVAILABLE:
                print(f"Skipping quantum seed={s}: Qiskit/Torch no disponibles en este entorno.")
                continue
            tasks.append((a, s))

    if not tasks:
        print("No hay tareas para ejecutar. Revisa --agents y disponibilidad de Qiskit.")
        return

    # parámetros (ajustar aquí si QNN simula muy lento)
    game_kwargs = {'max_life': 3, 'height_px': 40, 'width_px': 50, 'movimiento_px': 3, 'default_reward': args.reward}
    classical_kwargs = {'discount_factor':0.2, 'learning_rate':0.1, 'ratio_explotacion':0.85}
    quantum_kwargs = {'discount_factor':0.2, 'learning_rate':0.01, 'ratio_explotacion':0.85, 'num_qubits':3, 'reps':1}

    # seleccionar kwargs por tarea en el worker: run_single recibe agent_kwargs, game_kwargs
    # Lanzar en paralelo
    print(f"Lanzando {len(tasks)} tareas en {args.workers} workers...")
    # Mapear agent type a kwargs por cada tarea antes de enviar
    # (hacemos esto localmente y el worker recibe un agent_kwargs concreto)
    prepared_tasks = []
    for (a, s) in tasks:
        ak = classical_kwargs.copy() if a == 'classical' else quantum_kwargs.copy()
        prepared_tasks.append((a, s, ak, game_kwargs))

    # Ejecutar en paralelo: adaptamos run_tasks_in_parallel para recibir kwargs
    # Construimos una wrapper simple que llama run_single con los kwargs — pero run_single already accepts them
    # We'll use a small helper to submit correct parameters.
    tasks_simple = [(a, s) for (a, s, ak, gk) in prepared_tasks]

    # Instead of duplicating data, create a mapping seed->agent_kwargs for retrieval in child via run_single args
    # But simpler: submit each with its own agent_kwargs
    classical_results = []
    quantum_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_to_meta = {}
        for (a, s, ak, gk) in prepared_tasks:
            fut = ex.submit(run_single, a, args.mode, args.budget, s, ak, gk)
            fut_to_meta[fut] = (a, s)
        for fut in concurrent.futures.as_completed(fut_to_meta):
            a, s = fut_to_meta[fut]
            try:
                res = fut.result()
                if res['agent_type'] == 'classical':
                    classical_results.append(res)
                else:
                    quantum_results.append(res)
                print(f"Finished: {a} seed={s} eps={res['episodes_done']} time={res['total_time_s']:.1f}s avg_r={res['avg_reward_per_ep']:.2f}")
            except Exception:
                print(f"Task failed: {a} seed={s}. Traceback:")
                traceback.print_exc()

    # ordenar por seed
    classical_results.sort(key=lambda x: x['seed'])
    quantum_results.sort(key=lambda x: x['seed'])

    # resumen e impresión
    if classical_results:
        df_classic = aggregate_and_print(classical_results, "CLÁSICO")
        df_classic.to_csv(os.path.join(args.out, "classical_summary.csv"), index=False)
        save_detail(classical_results, os.path.join(args.out, "classical_rewards.csv"))

    if quantum_results:
        df_quantum = aggregate_and_print(quantum_results, "CUÁNTICO")
        df_quantum.to_csv(os.path.join(args.out, "quantum_summary.csv"), index=False)
        save_detail(quantum_results, os.path.join(args.out, "quantum_rewards.csv"))

    # plot ejemplo (reward medio por episodio across seeds)
    try:
        fig, ax = plt.subplots()
        if classical_results:
            dfc = pd.read_csv(os.path.join(args.out, "classical_rewards.csv"))
            gp = dfc.groupby('ep')['reward'].agg(['mean','std']).reset_index()
            ax.plot(gp['ep'], gp['mean'], label='clásico')
        if quantum_results:
            dfq = pd.read_csv(os.path.join(args.out, "quantum_rewards.csv"))
            gq = dfq.groupby('ep')['reward'].agg(['mean','std']).reset_index()
            ax.plot(gq['ep'], gq['mean'], label='cuántico')
        ax.set_xlabel('episodio')
        ax.set_ylabel('reward')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "learning_curves.png"))
        print(f"Gráfica guardada en {args.out}/learning_curves.png")
    except Exception as e:
        print("No se pudo generar la gráfica:", e)


if __name__ == '__main__':
    main()
