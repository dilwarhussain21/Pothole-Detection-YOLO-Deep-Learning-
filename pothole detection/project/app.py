import cv2
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
from delete import *


from ultralytics import YOLO

app = Flask(__name__)
try:
    deletefiles()
except:
    pass
@app.route('/')
def index():
    return render_template('mainpage.html')

@app.route("/", methods=["GET","POST"])
def predict_img():
    if request.method=="POST":
        if 'file' in request.files:
            f=request.files['file']
            basepath=os.path.dirname(__file__)
            filepath=os.path.join(basepath,'uploads',f.filename)
            print("Upload folder is ",filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath=f.filename

            print("printing predict_img ::::::::",predict_img)

            file_extension=f.filename.rsplit('.',1)[1].lower()
            print('file extension = '+file_extension)

            if file_extension == 'jpg' or file_extension =='png' or file_extension== 'jpeg':
                img = cv2.imread(filepath)

                yolo=YOLO('pothole.pt')
                detections=yolo.predict(img, save=True)
                return display(f.filename)
            
            elif file_extension == 'mp4':
                video_path=filepath
                print('Video path'+video_path)
                cap = cv2.VideoCapture(video_path) #it opens the video file and look for parameters.

                #to get the dimension of the video.
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc=cv2.VideoWriter_fourcc(*'mp4v')  
                out=cv2.VideoWriter('output.mp4', fourcc, 30.0,(frame_width , frame_height))

                model = YOLO('pothole.pt')
                # detections=model.predict(video_path, vid_stride= True, save=True, show=True)

                while cap.isOpened():
                    ret, frame =cap.read()
                    if not ret:
                        break


                    results= model(frame ,save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted=results[0].plot()
                    cv2.imshow("results",res_plotted)

                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break




# The display function is used to serve the image from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    files=os.listdir(directory)
    latest_file=files[0]

    print(latest_file)

    filename=os.path.join(folder_path,latest_subfolder,latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg' or 'png':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab


    else:

        return "Invalid file format"


if __name__ == '__main__':
    app.run()


