import numpy as np
import matplotlib.pyplot as plt

alg = ['HUGO', 'LSB', 'WOW', 'SUNIWARD']
i_path = [0.1, 0.2, 0.3, 0.4]

for i in i_path:
    for i2 in alg:
        file_path = f'{i}/Yedroudj_{i2}_5000bosstrain_1500valid_no_DA_v1.npz'

        data = np.load(file_path)

        loss_train_log = data['loss_train_log']
        DAC_train_log = data['DAC_train_log']
        stego_AC_train_log = data['stego_AC_train_log']
        cover_AC_train_log = data['cover_AC_train_log']
        cover_AC_valid_log = data['cover_AC_valid_log']
        stego_AC_valid_log = data['stego_AC_valid_log']
        DAC_valid_log = data['DAC_valid_log']

        data.close()

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))

        axes[0, 0].plot(loss_train_log, label='Training Loss', color='blue')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(DAC_train_log, label='Training DAC', color='green')
        axes[0, 1].set_title('Training DAC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('DAC')
        axes[0, 1].set_ylim(50, 100)
        axes[0, 1].legend()

        axes[1, 0].plot(DAC_valid_log, label='Validation DAC', color='red')
        axes[1, 0].set_title('Validation DAC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('DAC')
        axes[1, 0].set_ylim(50, 100)
        axes[1, 0].legend()

        axes[1, 1].plot(cover_AC_train_log, label='Cover AC (train)', color='red')
        axes[1, 1].plot(stego_AC_train_log, label='Stego AC (train)', color='green')
        axes[1, 1].set_title('Cover and Stego AC (Train)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AC')
        axes[1, 1].set_ylim(50, 100)
        axes[1, 1].legend()

        axes[2, 0].plot(cover_AC_valid_log, label='Cover AC (valid)', color='red')
        axes[2, 0].set_title('Cover AC (Valid)')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('AC')
        axes[2, 0].set_ylim(0, 100)
        axes[2, 0].legend()

        axes[2, 1].plot(stego_AC_valid_log, label='Stego AC (valid)', color='purple')
        axes[2, 1].set_title('Stego AC (Valid)')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('AC')
        axes[2, 1].set_ylim(0, 100)
        axes[2, 1].legend()

        fig.suptitle(f'{i}_{i2} Results', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(f'{i}_{i2}_result.png')
