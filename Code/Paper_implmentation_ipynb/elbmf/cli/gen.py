import numpy as np
matrix = np.random.randint(2, size=(100, 150), dtype=np.int8)
np.savetxt("matrix.tsv", matrix, delimiter="\t", fmt="%d")
