import cv2
import numpy as np
import matplotlib.pyplot as plt
alg = ['HUGO', 'LSB', 'WOW', 'SUNIWARD']
i_path = [0.1, 0.2, 0.3, 0.4]

for i in i_path:
    for i2 in alg:
        file_path = f'{i}_{i2}_result.png'
        image = cv2.imread(file_path)
        image_numpy = np.array(image)
        crop_image = image_numpy[100:1122, 100:1400, :]
        plt.imshow(crop_image)
        # plt.show()
        cv2.imwrite(f'{i}/{i2}result.jpg', crop_image)