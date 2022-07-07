import h5py
import numpy as np
import sys

file1 = sys.argv[-2]
file2 = sys.argv[-1]

df1 = h5py.File(file1, 'r')
t1 = df1['scales/sim_time'][0]
b1 = df1['tasks/b'][0]
df1.close()

df2 = h5py.File(file2, 'r')
t2 = df2['scales/sim_time'][0]
b2 = df2['tasks/b'][0]
df2.close()

delta = b1 - b2
print(f"t1 = {t1:e}, t2 = {t2:e}")
print(f"||b1 - b2||_F = {np.linalg.norm(delta):e}")
