import numpy as np
import pickle
import torch
import io
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


if __name__ == '__main__':

    path = 'out.p'

    with open(path, 'rb') as file:
        out_dict = CPU_Unpickler(file).load()

    print()