import torch
from PIL import Image
import torch.nn as nn
import numpy as np

pic=Image.open('test1.png')
pic_array=np.asarray(pic)
print('pic_array.shape:',pic_array.shape)
pic_array_re=pic_array.reshape(1,1,280,280)
print('pic_array.shape_re:',pic_array_re.shape)
pic_tensor=torch.Tensor(pic_array_re)
print('pic_tensor.shape:',pic_tensor.shape)
Cov=nn.Conv2d(1,1,4,1,0) #(inchannels,out_channels,kernel_size,stride,padding)
pic_cov=Cov(pic_tensor)
print('pic_cov.shape:',pic_cov.shape)


Cov_tran=nn.ConvTranspose2d(1,1,4,1,0)
pic_cov_tran=Cov_tran(pic_cov)
print('pic_cov_tran.shape:',pic_cov_tran.shape)