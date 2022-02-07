#Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdetection.mmdet as mmdet
print(mmdet.__version__)
from mmdetection.mmdet.apis import inference_detector, init_detector, show_result_pyplot

import matplotlib
matplotlib.pyplot.switch_backend('Agg')

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


config = 'mmdetection/configs/ssd/ssd300_coco.py'
checkpoint = 'ssd300_coco_20210803_015428-d231a06e.pth'
img = 'mmdetection/demo/demo.jpg'
model = init_detector(config, checkpoint, device='cpu')
result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.3)

#TODO Больше картинок для тестов

import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

WAS_FOLDER = os.path.join('static', 'was')
RESULT_FOLDER = os.path.join('static', 'result')

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['WAS_FOLDER'] = WAS_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            upload_path = os.path.join(app.config['WAS_FOLDER'], filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'],filename.rsplit('.', 1)[0].lower() + "_res." + filename.rsplit('.', 1)[1].lower())

            print(upload_path)
            print(result_path)

            file.save(upload_path)

            result = inference_detector(model, upload_path)
            print(len(result))
            model.show_result(upload_path,
                result,
                score_thr=0.3,
                show=True,
                wait_time=0,
                win_name='result',
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                out_file=result_path)
            return render_template('show_result.html', was=upload_path, result=result_path)
        else:
            flash('Allowed extensions: png,  jpg, jpeg')
            return redirect(request.url)




    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0',host=80)
