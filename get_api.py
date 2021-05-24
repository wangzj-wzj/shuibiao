import requests,json
import time
import cv2
import skimage.io
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#data = skimage.io.imread('/home/wangzj/WORK/kaiguan/csv/data/001411.jpg')  # picture data
#data = cv2.imread('/home/wangzj/WORK/shuibiao/CNN_shuibiao/testdata1/10.png')
TOKEN_URL = 'http://0.0.0.0:8868/img_detect' # URL of server
st = time.time() # check time baseline
img_path = {'img_path':"/home/wangzj/WORK/shuibiao/CNN_shuibiao/testdata1/10.png"}
#img_array = json.dumps({"img_array":data},cls=NpEncoder) # transfer format from ndarray to json
result = requests.post(TOKEN_URL,data=img_path).json() 
print('Elapsed time: {}'.format(time.time()-st))
print (result)
