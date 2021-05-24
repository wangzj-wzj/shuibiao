import cv2
import os,shutil
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

def lapla(fpath):
    img = cv2.imread(fpath)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(grey, cv2.CV_64F).var()
    return imageVar

def read_data(data_dir):
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)

    return fpaths

def brenner(fpath):
    img = cv2.imread(fpath)
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)
    f = np.matrix(grey)
    tmp = filters.sobel(f)
    source=np.sum(tmp**2)
    source=np.sqrt(source)
    return source

if __name__ == "__main__":
#    fpath = '/home/wangzj/WORK/shuibiao/CNN_shuibiao/testdata2/0_0_8_4_4.1233.jpg'
#    var = lapla(fpath)
#    print(var)
    fpaths = read_data("./testdata/")
    out = open("lapla.txt",'w')
    newpath="./data_mohu"
    for fpath in fpaths:
        var = lapla(fpath)
#        print(var)
        if 800>var >700:
            out.write(fpath+'\t'+str(var)+'\n')
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            shutil.copy(fpath,newpath)


#  matplotlib.axes.Axes.hist() 方法的接口
'''
n, bins, patches = plt.hist(x=resu_var, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
plt.show()
'''
