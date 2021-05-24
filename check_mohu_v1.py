import cv2
import os
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    fpaths = read_data("./wrong_pre_pic")
    out = open("varlower500.txt",'w')
    resu_var = []
    for fpath in fpaths:
        var = lapla(fpath)
        resu_var.append(var)
#    if var < 300:
#        out.write(fpath+'\n')


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
