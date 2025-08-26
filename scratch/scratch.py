import torch
import numpy as np
import time
import os

cur_dir = os.path.dirname(__file__)

# create a 50% sparse matrix A that would be multiplied by B to produce C
np.random.seed(42)  # Set seed for reproducibility
M = 4096
K = 1024
N = 4096

A = np.random.randn(M, K)
B = np.random.randn(K, N)
# make A 70% sparse, rowwise
half = int(A.shape[1] * 0.9)
A = torch.tensor(A, dtype=torch.float16).to('cuda')
B = torch.tensor(B, dtype=torch.float16).to("cuda")
sort_res = torch.sort(A, dim=-1, stable=True)
indices = sort_res.indices
A.scatter_(1, indices[:, :half], 0)

t0 = time.time()
iter = 50
for i in range(iter):
    C = A @ B
t = (time.time()-t0)/iter
flops = 2*M*N*K
teraflop = flops/1e12
tflops = teraflop/t

print(f"Avg Time taken for 50 runs: {t:.4f} seconds")
print(f"TFLOPS: {tflops:.4f}")

# save A, B, and C to disk as binary to be loaded in c/c++ code
A.detach().cpu().numpy().tofile(f'{cur_dir}/A.bin')
# save the column index and row pointer for the sparse matrix A
A_sparse = A.to_sparse_csr()
A_sparse.values().detach().cpu().numpy().tofile(f'{cur_dir}/A_values.bin')
A_sparse.col_indices().to(torch.int32).detach().cpu().numpy().tofile(f'{cur_dir}/A_col_indices.bin')
A_sparse.crow_indices().to(torch.int32).detach().cpu().numpy().tofile(f'{cur_dir}/A_row_ptr.bin')
B.detach().cpu().numpy().tofile(f'{cur_dir}/B.bin')
C.detach().cpu().numpy().tofile(f'{cur_dir}/C.bin')


print(C.detach().cpu().numpy().ravel().tolist()[:10])
# breakpoint()