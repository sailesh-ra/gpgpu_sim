import numpy as np
import os
import sys
# =========================
# Configure here
# =========================


if len(sys.argv) != 7:
    print("Usage: python matgen.py X M N K layoutA layoutB")
    sys.exit(1)

X = int(sys.argv[1])
M = int(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])
layoutA = sys.argv[5]
layoutB = sys.argv[6]
seed = 42
# =========================

np.random.seed(seed)

# Create folder
folder_name = f"sample_{X}"
os.makedirs(folder_name, exist_ok=True)

# Generate matrices
A = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
B = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
C = np.matmul(A, B).astype(np.float32)


def write_matrix(path, matrix, layout):
    rows, cols = matrix.shape

    if layout == 'T':
        flat = matrix.flatten(order='C')
    elif layout == 'N':
        flat = matrix.flatten(order='F')
    else:
        raise ValueError("Layout must be 'T' or 'N'")

    with open(path, 'w') as f:
        f.write(f"{layout}\n")
        f.write(f"{rows}\n")
        f.write(f"{cols}\n")
        for val in flat:
            f.write(f"{val:.4f} ")


# File names
A_filename = f"A_{M}_{K}_{layoutA}.txt"
B_filename = f"B_{K}_{N}_{layoutB}.txt"
C_filename = f"C_{M}_{N}_T.txt"

# Full paths
A_path = os.path.join(folder_name, A_filename)
B_path = os.path.join(folder_name, B_filename)
C_path = os.path.join(folder_name, C_filename)

# Write files
write_matrix(A_path, A, layoutA)
write_matrix(B_path, B, layoutB)
write_matrix(C_path, C, 'T')

print(f"Generated folder: {folder_name}")
print(f"  {A_filename}")
print(f"  {B_filename}")
print(f"  {C_filename}")
