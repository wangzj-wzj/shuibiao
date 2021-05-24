from flask import Flask, render_template
from flask import request
import json
import os
import shutil
import sys
import pandas as pd
sys.path.append(r'/home/wangzj/WORK/shuibiao/CNN_shuibiao/')
import predict

PEOPLE_FOLDER = os.path.join('static','images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER 

def read_data(data_dir):
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
    return fpaths
 
@app.route('/img_detect', methods=["GET"])
def form():
    return render_template("form.html")

@app.route('/img_detect', methods=[ "POST"])
def detect():
#    if request.method == "POST":
#    img = request.form.get("img_path")
    img = request.form['img_path']
    fpaths = read_data(img)
    if len(fpaths) < 100:
        detect_results = []
        label_list = []
        num_list = []
        for fpath in fpaths:
            label = fpath.split('/')[-1].split('.')[0].split('_')
            resu = predict.predict_path(fpath) #run predict.py
            if ('检测数值' in resu):
                label_list.extend(label)
                num_list.extend(resu["检测数值"])
            detect_results.append(resu)
        index = 0
        for i in range(len(num_list)):
            if (label_list[i] != num_list[i]):
                index += 1
        acc = str(round(1-(index/len(num_list)),2)*100)+'%'
        for i in detect_results:
            for j in range(len(i['检测数值'])):
                if (int(i['检测数值'][j]) >=10) :
                    i['检测数值'][j] = str(int(i['检测数值'][j])-10+0.5)
            i['检测数值'] = ' '.join(i['检测数值'])
            shutil.copy(i['图片'].split('\"')[1], './static/images')
            i['图片'] = '<img src= "'+os.path.join(app.config['UPLOAD_FOLDER'],i['图片'].split('\"')[1].split('/')[-1])+'" />'
        df = pd.DataFrame.from_dict(detect_results)
        HEADER = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>水表数值检测</title>
        <link rel=stylesheet type=text/css href="{{ url_for('static', filename= 'style.css') }}">
    </head>
    <style type="text/css">
    .divForm{
        position: absolute;/*绝对定位*/
        width: 800px;
        height: 200px;
        text-align: center;/*(让div中的内容居中)*/
        top: 50%;
        left: 50%;
        margin-top: -300px;
        margin-left: -400px;
    }
    </style>

    <body>
    <div class="divForm">
        <h1>水表数据检测</h1>
        <h2>全部图像检测正确率:</h2>
        <font color = 'red'>{{ acc|safe }}</font>
        <h2>检测表:</h2>

    '''
        DF2HTML = df.to_html(classes='shuibiao',index=False)
        DF2HTML = DF2HTML.replace('&lt;','<')
        DF2HTML = DF2HTML.replace('/&gt;','/>')
        FOOTER = '''
    </div>
    </body>
    </html>
    '''

        with open('./templates/detect2.html','w') as f:
            f.write(HEADER)
            f.write(DF2HTML)
            f.write(FOOTER)
        return render_template("detect2.html", acc = acc)

    if len(fpaths) >= 100:
        detect_results = []
        label_list = []
        num_list = []
        for fpath in fpaths:
            label = fpath.split('/')[-1].split('.')[0].split('_')
            resu = predict.predict_path(fpath) #run predict.py
            if ('检测数值' in resu):
                label_list.extend(label)
                num_list.extend(resu["检测数值"])
            if ('图像问题' in resu):
                detect_results.append(resu)
        index = 0
        for i in range(len(num_list)):
            if (label_list[i] != num_list[i]):
                index += 1
        acc = str(round(1-(index/len(num_list)),2)*100)+'%'
        for i in detect_results:
            shutil.copy(i['图片'].split('\"')[1], './static/images')
            i['图片'] = '<img src= '+os.path.join(app.config['UPLOAD_FOLDER'],i['图片'].split('\"')[1].split('/')[-1])+'/>'
            i['图像问题'] = '<font color = \'red\'>图片模糊</font>，建议重录'
        df = pd.DataFrame.from_dict(detect_results)
        HEADER = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>水表数值检测</title>
        <link rel=stylesheet type=text/css href="{{ url_for('static', filename= 'style.css') }}">
    </head>
    <style type="text/css">
    .divForm{
        position: absolute;/*绝对定位*/
        width: 800px;
        height: 200px;
        text-align: center;/*(让div中的内容居中)*/
        top: 50%;
        left: 50%;
        margin-top: -300px;
        margin-left: -400px;
    }
    </style>

    <body>
    <div class="divForm">
        <h1>水表数据检测</h1>
        <h2>检测图片数量:</h2>
        {{ num_pic|safe }}
        <h2>除不合格图像以外图像检测正确率:</h2>
        <font color = 'red'>{{ acc|safe }}</font>
        <h2>不合格图像列表:</h2>

    '''
        DF2HTML = df.to_html(classes='shuibiao',index=False)
        DF2HTML = DF2HTML.replace('&lt;','<')
        DF2HTML = DF2HTML.replace('/&gt;','/>')
        FOOTER = '''
    </div>
    </body>
    </html>
    '''

        with open('./templates/detect_large.html','w') as f:
            f.write(HEADER)
            f.write(DF2HTML)
            f.write(FOOTER)
        return render_template("detect_large.html", acc = acc, num_pic = len(fpaths))



 
if __name__ == '__main__':
    app.run(port=8808, host='0.0.0.0',debug=True)
