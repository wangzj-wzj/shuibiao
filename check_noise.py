import cv2
from PIL import Image
from PIL import ImageChops
import numpy as np
import time
import pytesseract
import warnings

warnings.filterwarnings("ignore")
demo=Image.open("./varlower500/0_0_0_8_0_16.1313.png")
im=np.array(demo.convert('L'))#灰度化矩阵
print(im.shape)
print(im.dtype)
#print(im)
height=im.shape[0]#尺寸
width=im.shape[1]
varlist=[]
for i in range(height):
    for j in range(width):
        for k in range(16):
            if im[i][j]>=k*16 and im[i][j]<(k+1)*16:#16级量化
                im[i][j]=8*(k*2+1)
                break
for i in range(0,height-height%3,3):
    for j in range(0,width-width%3,3):
        x=(im[i][j]+im[i+1][j]+im[i+2][j]+im[i][j+1]+im[i+1][j+1]+im[i+2][j+1]+im[i][j+2]+im[i+1][j+2]+im[i+2][j+2])/9
        x2=(pow(im[i][j],2)+pow(im[i+1][j],2)+pow(im[i+2][j],2)+pow(im[i][j+1],2)+pow(im[i+1][j+1],2)+pow(im[i+2][j+1],2)+pow(im[i][j+2],2)+pow(im[i+1][j+2],2)+pow(im[i+2][j+2],2))/9
        var=x2-pow(x,2)
        varlist.append(round(var,3))#子窗口的方差值3x3
print(im)
#print(varlist)
T=round(sum(varlist)/len(varlist),3)#保留3位小数
print(T)
