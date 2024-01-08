import numpy as np
import matplotlib.pyplot as plt

alg = ['HUGO', 'LSB', 'WOW', 'SUNIWARD']
i_path = [0.1, 0.2, 0.3, 0.4]

for i in i_path:
    for i2 in alg:
        file_path = f'New folder (2)/{i}/Yedroudj_{i2}_5000bosstrain_1500valid_no_DA_v1.npz'

        data = np.load(file_path)

        loss_train_log = data['loss_train_log']
        DAC_train_log = data['DAC_train_log']
        stego_AC_train_log = data['stego_AC_train_log']
        cover_AC_train_log = data['cover_AC_train_log']
        cover_AC_valid_log = data['cover_AC_valid_log']
        stego_AC_valid_log = data['stego_AC_valid_log']
        DAC_valid_log = data['DAC_valid_log']

        print(f'{i} , {i2} = {np.max(DAC_valid_log[:200])})')



