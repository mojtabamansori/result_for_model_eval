import numpy as np
import matplotlib.pyplot as plt
alg = ['HUGO', 'LSB', 'WOW', 'SUNIWARD']
i_path = [0.1, 0.2, 0.3, 0.4]
for i in i_path:
    for i2 in alg:
        file_path = f'{i}/Yedroudj_{i2}_5000bosstrain_1500valid_no_DA_v1.npz'

        # Load the .npz file
        data = np.load(file_path)

        # Extract data
        loss_train_log = data['loss_train_log']
        DAC_train_log = data['DAC_train_log']
        DAC_valid_log = data['DAC_valid_log']

        # Close the file
        data.close()

        plt.figure(figsize=(10, 8))
        plt.subplot(311)
        plt.plot(loss_train_log, label='Training Loss', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(DAC_train_log, label='Training DAC', color='green')
        plt.title('Training DAC')
        plt.xlabel('Epoch')
        plt.ylabel('DAC')
        plt.ylim(50,100)
        plt.legend()
        plt.subplot(313)
        plt.plot(DAC_valid_log, label='Validation DAC', color='red')
        plt.title('Validation DAC')
        plt.xlabel('Epoch')
        plt.ylabel('DAC')
        plt.ylim(50,100)
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.savefig(f'{i}_{i2}_result.png')
