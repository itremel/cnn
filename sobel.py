import matplotlib.pyplot as plt
import numpy as np
sobel=np.array([[1,  2, 0,  -2, -1],
[4,  8, 0,  -8, -4],
[6, 12, 0, -12, -6],
[4,  8, 0,  -8, -4],
[1,  2, 0,  -2, -1]])

ysobel=np.array([[1,  4, 6,  4, 1],
[2,  8, 12,  8, 2],
[0, 0, 0, 0, 0],
[-2,  -8, -12,  -8, -2],
[-1,  -4, -6,  -4, -1]])

print(sobel.shape)
plt.imsave('xsobel.svg',sobel)
plt.imsave('ysobel.svg',ysobel)
#plt.imshow(sobel)
#plt.show()    
