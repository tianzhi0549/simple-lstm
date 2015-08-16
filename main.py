from simple_lstm import SimpleLSTM
import numpy as np
input_dim=50
mem_cell_cnt=100
lstm=SimpleLSTM(mem_cell_cnt, input_dim, 1)
input_sequence=np.load("input.npy")
lstm.fit(input_sequence, [_[0] for _ in input_sequence])
