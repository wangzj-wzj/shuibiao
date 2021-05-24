import flask, json
from flask import request
import predict 
import numpy as np

'''
flask： web框架，通过flask提供的装饰器@server.route()将普通函数转换为服务
'''
# 创建一个服务，把当前这个python文件当做一个服务
server = flask.Flask(__name__)
# server.config['JSON_AS_ASCII'] = False
# @server.route()可以将普通函数转变为服务 登录接口的路径、请求方式
@server.route('/img_detect', methods=['get', 'post']) #page defination
def img_detect():
    # 获取通过url请求传参的数据
#    json_dump = request.get_json('img_array')  #get parameter in json format
#    img = np.asarray(json_dump["img_array"])  #transfer to ndarray format
    img =  request.form['img_path']
    detect_results = predict.predict_path(img)  #run predict.py
    return json.dumps(detect_results, ensure_ascii=False)  # 将字典转换为json串, json是字符串
 
if __name__ == '__main__':
    server.run(debug=True, port=8868, host='0.0.0.0')# 指定端口、host,0.0.0.0代表不管几个网卡，任何ip都可以访问
