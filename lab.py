from torchvision import datasets, transforms
#datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
#minst=datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
# from PIL import Image
# from torchvision import transforms
# import numpy as np
# crop=transforms.Scale(12)#将输入的`PIL.Image`重新改变大小成给定的`size`，`size`是最小边的边长。举个例子，如果原图的`height>width`,那么改变大小后的图片大小是`(size*height/width, size)`，若是height<width,那么就是(size, size*width/height)。
# img=Image.open('test.jpg')
# print(type(img))
# print(np.asarray(img))
# print(crop(img).size)


from PIL import Image
from torchvision import transforms
import numpy as np
im = Image.open('test.jpg')
print(im)
com=transforms.Compose([
     transforms.Resize((3,4)),
     transforms.ToTensor(),
 ])
im_com=com(im)
print(im_com)

