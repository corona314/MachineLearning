import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar los datos ---
df_classical = pd.read_csv("new_qnn_500epi\\classical_rewards.csv")
df_quantum   = pd.read_csv("new_qnn_500epi\\quantum_rewards.csv")

block_size = 50

# --- Función para calcular medias por bloque ---
def block_means(df, block_size):
    df['block'] = df['ep'] // block_size
    return df.groupby('block')['reward'].mean()

classical_blocks = block_means(df_classical, block_size)
quantum_blocks   = block_means(df_quantum, block_size)
common_blocks = classical_blocks.index.intersection(quantum_blocks.index)

# --- Comparación bloque a bloque ---
better_classical = 0
better_quantum   = 0

print("Comparación por bloques (reward medio):")
for block in common_blocks:
    c_mean = classical_blocks[block]
    q_mean = quantum_blocks[block]
    if c_mean > q_mean:
        winner = "Classical"
        better_classical += 1
    elif q_mean > c_mean:
        winner = "Quantum"
        better_quantum += 1
    else:
        winner = "Empate"
    print(f"Bloque {block*block_size:03d}-{block*block_size+block_size-1:03d}: Classical={c_mean:.2f}, Quantum={q_mean:.2f} -> {winner}")

# --- Resultado global ---
print("\nResumen global:")
if better_classical > better_quantum:
    print(f"Classical gana en más bloques ({better_classical} vs {better_quantum})")
elif better_quantum > better_classical:
    print(f"Quantum gana en más bloques ({better_quantum} vs {better_classical})")
else:
    print(f"Empate global ({better_classical} vs {better_quantum})")

# --- Gráfica ---
plt.plot(classical_blocks.index*block_size, classical_blocks.values, label="Classical", marker='o')
plt.plot(quantum_blocks.index*block_size, quantum_blocks.values, label="Quantum", marker='o')
plt.xlabel("Episodio")
plt.ylabel("Reward medio por bloque")
plt.title("Comparación Classical vs Quantum")
plt.legend()
plt.grid(True)
plt.show()
