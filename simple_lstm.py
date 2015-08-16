# coding=utf8
import numpy as np


def rand_arr(a, b, *args):
    return np.random.rand(*args) * (b - a) + a


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class SimpleLSTM:
    def __init__(self, mem_cell_cnt, input_dim, output_dim):
        # 权重矩阵，每一行表示一个记忆细胞的一个输入点的所有权重。
        # 比如说Wg就表示g这个输入点。输入点是g, input gate, output gate, forget gate的统称。
        # 因为总计有mem_cell_cnt个记忆细胞。
        # 每个记忆细胞的每个输入点要和所有其他的记忆细胞的上一时刻的输出以及本时刻的输入相连。
        # 所以每个细胞的每个输入点就需要mem_cell_cnt + input_dim个权重。
        # 因为总计有mem_cell_cnt个记忆细胞，所以是mmem_cell_cnt行。
        self.Wg = rand_arr(-0.1,  0.1, mem_cell_cnt, mem_cell_cnt + input_dim)
        self.Wi = rand_arr(-0.1,  0.1, mem_cell_cnt, mem_cell_cnt + input_dim)
        self.Wo = rand_arr(-0.1,  0.1, mem_cell_cnt, mem_cell_cnt + input_dim)
        self.Wf = rand_arr(-0.1,  0.1, mem_cell_cnt, mem_cell_cnt + input_dim)
        # 最后所有的隐藏层连接到一个输出层的权重。
        self.Wu = rand_arr(-0.1,  0.1, output_dim, mem_cell_cnt)
        self.bu = rand_arr(-0.1,  0.1, output_dim)
        # 每个细胞的每个输入点有一个偏置项。
        self.bg = rand_arr(-0.1,  0.1, mem_cell_cnt)
        self.bi = rand_arr(-0.1,  0.1, mem_cell_cnt)
        self.bo = rand_arr(-0.1,  0.1, mem_cell_cnt)
        self.bf = rand_arr(-0.1,  0.1, mem_cell_cnt)
        # 所有细胞上一时刻的输出。
        self.prev_h = np.zeros(mem_cell_cnt)
        # 所有细胞上一时刻的内部状态s.
        self.prev_s = np.zeros(mem_cell_cnt)
        # 最终的loss想对于每个变量的导数。
        self.Wg_diff = np.zeros_like(self.Wg)
        self.Wi_diff = np.zeros_like(self.Wi)
        self.Wo_diff = np.zeros_like(self.Wo)
        self.Wf_diff = np.zeros_like(self.Wf)
        self.Wu_diff = np.zeros_like(self.Wu)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bo_diff = np.zeros_like(self.bo)
        self.bf_diff = np.zeros_like(self.bf)
        self.bu_diff = np.zeros_like(self.bu)
        # 训练时系统当前的输入输出。
        self._input_sequence = None
        self._output_sequence = None

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        mem_cell_cnt = len(data[0])
        input_dim = len(data[0][0]) - mem_cell_cnt
        instance = cls(mem_cell_cnt, input_dim)
        instance.Wg = np.vstack(data[0])
        instance.Wi = np.vstack(data[1])
        instance.Wo = np.vstack(data[3])
        instance.Wf = np.vstack(data[2])
        instance.bg = np.hstack(data[4])
        instance.bi = np.hstack(data[5])
        instance.bo = np.hstack(data[7])
        instance.bf = np.hstack(data[6])
        return instance

    def save(self, file_name):
        np.save(file_name, [self.Wg, self.Wi, self.Wf, self.Wo,
                            self.bg, self.bi, self.bf, self.bo])

    def fit(self, input_sequence, output_sequence):
        assert len(input_sequence) == len(output_sequence)
        self._input_sequence = np.array(input_sequence)
        self._output_sequence = np.array(output_sequence)
        self.compute_Wu_bu_diff()

    def compute_Wu_bu_diff(self):
        self.reset()
        for t in range(len(self._input_sequence)):
            u_minus_y = self._predict(self._input_sequence[t]) - self._output_sequence[t]
            diff = np.asmatrix(u_minus_y).T * \
                np.asmatrix(self.prev_h)
            self.Wu_diff += np.asarray(diff)
            self.bu_diff += u_minus_y

    def diff_l(self, t):
        pass

    def predict(self, input_sequence):
        r = []
        self.reset()
        for _input in input_sequence:
            r.append(self._predict(_input))
        return r

    def _predict(self, x):
        concat_input = np.hstack((x, self.prev_h))
        g = np.tanh(np.dot(self.Wg, concat_input) + self.bg)
        i = sigmoid(np.dot(self.Wi, concat_input) + self.bi)
        o = sigmoid(np.dot(self.Wo, concat_input) + self.bo)
        f = sigmoid(np.dot(self.Wf, concat_input) + self.bf)
        s = g*i + self.prev_s*f
        h = s*o
        self.prev_s = s
        self.prev_h = h
        return np.dot(self.Wu, h)+self.bu

    def reset(self):
        # 所有细胞上一时刻的输出。
        self.prev_h = np.zeros_like(self.prev_h)
        # 所有细胞上一时刻的内部状态s.
        self.prev_s = np.zeros_like(self.prev_s)
