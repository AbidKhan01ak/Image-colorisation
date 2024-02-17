from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np 
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\Users\\akstu\\OneDrive\\Desktop\\Image colorisation\\uploads'
RESULT_FOLDER = 'C:\\Users\\akstu\\OneDrive\\Desktop\\Image colorisation\\results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v0.caffemodel')
pts = np.load('pts_in_hull.npy')

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

def colorize_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32")/255.0
    lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab,(224,224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1,2,0))
    ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized,0,1)
    colorized = (255 * colorized).astype("uint8")

    filename = os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, colorized)

    return result_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        colorized_image_path = colorize_image(file_path)
        return render_template('result.html', original_image=filename, colorized_image=os.path.basename(colorized_image_path))
    
    return 'Invalid file format'

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
