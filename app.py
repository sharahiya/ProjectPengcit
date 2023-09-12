from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def histogram_equalization():
    if request.method == 'POST':
        #upload foto yg diinginkan
        picture = request.files['img']
        filename = secure_filename(picture.filename)
        picture.save(os.path.join(app.config['UPLOAD'], filename))
        img_address = os.path.join(app.config['UPLOAD'], filename)
        img = cv2.imread(img_address)
        
        
         # Hasil akhir foto yang equalisasi
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
        img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0]) 
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)

        equalized_image_address = os.path.join('static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_address, img_equalized)


        # Histogram utk gambar yg diupload
        R_histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
        G_histogram = cv2.calcHist([img], [1], None, [256], [0, 256])
        B_histogram = cv2.calcHist([img], [2], None, [256], [0, 256])

        R_histogram /= R_histogram.sum()
        G_histogram /= G_histogram.sum()
        B_histogram /= B_histogram.sum()

        hist_image_address = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(R_histogram, color='red', label='Red')
        plt.plot(G_histogram, color='green', label='Green')
        plt.plot(B_histogram, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_address)
        

        # Menghasilkan distribusi piksel setelah equalisasi gambar.
        hist_equalized_r = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist([img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist([img_equalized], [2], None, [256], [0, 256])
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()
       
        hist_equalized_image_address = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_address)

        return render_template('home.html', img=img_address, img2=equalized_image_address, histogram=hist_image_address, histogram2=hist_equalized_image_address)
    
    return render_template('home.html')


if __name__ == '__main__': 
    app.run(debug=True,port=8001)