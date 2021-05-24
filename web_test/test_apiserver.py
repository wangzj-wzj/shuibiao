from flask import Flask, g
from flask_restful import reqparse, Api, Resource
from flask_httpauth import HTTPTokenAuth
import predict
import json
import os

# Flask相关变量声明
app = Flask(__name__)
#api = Api(app)

# RESTfulAPI的参数解析 -- put / post参数解析
parser_put = reqparse.RequestParser()
parser_put.add_argument("img_path", type=str, required=True, help="image path")
#parser_put.add_argument("", type=str, required=True, help="")

def read_data(data_dir):
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
    return fpaths

# 操作（post / get）资源列表
@app.route('/img_detect', methods=['get', 'post'])
def img_detect():
    args = parser_put.parse_args()

        # 构建新参数
    img = args['img_path']
        # 调用function
    fpaths = read_data(img)
    detect_results = []
    label_list = []
    num_list = []
    for fpath in fpaths:
        label = fpath.split('/')[-1].split('.')[0].split('_')
        resu = predict.predict_path(fpath) #run predict.py
        if ('number' in resu[1]):
            label_list.extend(label)
            num_list.extend(resu[1]["number"])

        detect_results.append(resu)
    index = 0
    for i in range(len(num_list)):
        if (label_list[i] != num_list[i]):
            index += 1
    acc = 1-(index/len(num_list))
    detect_results.append(acc)
    return json.dumps(detect_results, ensure_ascii=False)  # 将字典转换为json串, json是字符串
        




if __name__ == "__main__":
    app.run(debug=True, port=8868, host='0.0.0.0')

