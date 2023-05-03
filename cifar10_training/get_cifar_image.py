import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:\HUJI\lab_weiss\truly_shift_invariant_cnns\cifar10_training\data\data_batch_1', 'rb') as f:
    imgList= pickle.load(f, encoding="bytes")

img = np.reshape(imgList[b"data"][0],(3,32,32)) # get the first element from list

# inorder to view in imshow we need image of type (height,width, channel) rather than (channel, height,width)
imgView=np.transpose(img, (1,2,0))

plt.imshow(imgView)
plt.show()
x = 5