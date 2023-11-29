import array
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from steganography import encode, decode, histogram, countn_byte

app = Flask(__name__)

class Cartoonizer: 
    """Cartoonizer effect 
    A class that applies a cartoon effect to an image. 
    The class uses a bilateral filter and adaptive thresholding to create 
    a cartoon effect. 
    """
    def __init__(self): 
        pass

    def render(self, img_rgb): 
        img_rgb = cv2.imread(img_rgb) 
        img_rgb = cv2.resize(img_rgb, (1366,768)) 
        numDownSamples = 2  # number of downscaling steps 
        numBilateralFilters = 100 # number of bilateral filtering steps 

        # -- STEP 1 -- 

        # downsample image using Gaussian pyramid 
        img_color = img_rgb 
        for _ in range(numDownSamples): 
            img_color = cv2.pyrDown(img_color) 

            #cv2.imshow("downcolor",img_color) 
            #cv2.waitKey(0) 
            # repeatedly apply small bilateral filter instead of applying 
            # one large filter 
            for _ in range(numBilateralFilters): 
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7) 

            #cv2.imshow("bilateral filter",img_color) 
            #cv2.waitKey(0) 
            # upsample image to original size 
            for _ in range(numDownSamples): 
                img_color = cv2.pyrUp(img_color) 
            #cv2.imshow("upscaling",img_color) 
            #cv2.waitKey(0) 

            # -- STEPS 2 and 3 -- 
            # convert to grayscale and apply median blur 
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 
            img_blur = cv2.medianBlur(img_gray, 3) 
            #cv2.imshow("grayscale+median blur",img_color) 
            #cv2.waitKey(0) 

            # -- STEP 4 -- 
            # detect and enhance edges 
            img_edge = cv2.adaptiveThreshold(img_blur, 255, 
                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, 9, 2) 
            #cv2.imshow("edge",img_edge) 
            #cv2.waitKey(0) 

            # -- STEP 5 -- 
            # convert back to color so that it can be bit-ANDed with color image 
            (x,y,z) = img_color.shape 
            img_edge = cv2.resize(img_edge,(y,x)) 
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB) 
            cv2.imwrite("edge.png",img_edge) 
            #cv2.imshow("step 5", img_edge) 
            #cv2.waitKey(0) 
            #img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2])) 
            #print img_edge.shape, img_color.shape 
            cartoonized_image = cv2.bitwise_and(img_color, img_edge)

        return cartoonized_image


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


def blurwajah(path_img, intensitas):
    img = cv2.imread(path_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=[25, 25])
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        kernel_size = (intensitas, intensitas)
        blurred_face = cv2.GaussianBlur(face, kernel_size, 0)
        img[y:y+h, x:x+w] = blurred_face

    gambar_blur = os.path.join(app.config['UPLOAD'], 'gambar_blur.jpg')
    cv2.imwrite(gambar_blur, img)

    return gambar_blur


@app.route('/secondpage', methods=['GET', 'POST'])
def bluredpage():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        path_img = os.path.join(app.config['UPLOAD'], filename)

        intensitas = int(request.form.get('tingkatan', 1))
        gambar_blur = blurwajah(path_img, intensitas)
        return render_template('dazzcam.html', img=path_img, fotoblur=gambar_blur)
    return render_template('dazzcam.html')

def edgefunction(img):
    edges = cv2.Canny(img, 150, 250) 

    gambar_edge = os.path.join(app.config['UPLOAD'], 'gambar_edge.jpg')
    cv2.imwrite(gambar_edge, edges)

    return gambar_edge

@app.route('/thirdpage', methods=['GET', 'POST'])
def edgedetection():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)
        gambar_edge = edgefunction(img)
        return render_template('edgedetection.html', image=img_path, edge=gambar_edge)
    return render_template('edgedetection.html')

def hapus(img_path):
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = cv2.GC_PR_BGD
    rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100) 

    cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_removed_bg = img * mask2[:, :, np.newaxis]
    
    removed_bg_image_path = os.path.join(app.config['UPLOAD'], 'hapus_background.jpg')
    cv2.imwrite(removed_bg_image_path, img_removed_bg)

    return removed_bg_image_path

@app.route('/segmentasi', methods=['GET', 'POST'])
def hapusbg():
    if request.method == 'POST':
        file = request.files['img']

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)
        remove_background_img = hapus(img_path)

        return render_template('hapus_background.html', img=img_path, img2=remove_background_img)

    return render_template('hapus_background.html')

@app.route('/Steganography', methods=['GET', 'POST'])
def stega():
    
    if request.method == 'POST':
        file = request.files['img']

        secret_data = request.form['text_input']

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)
        # remove_background_img = hapus(img_path)

# Encode the data into the image
        encoded_image = encode(image_name=img_path, secret_data=secret_data)
        output_image_path = os.path.splitext(img_path)[0] + '_encoded.png'
        cv2.imwrite(output_image_path, encoded_image)

        # Decode the secret data from the encoded image
        decoded_data = decode(output_image_path)
        input_plot, output_plot, percentage, comparison = histogram(input_image=img_path, output_image=output_image_path)

        return render_template('steganography.html', decoded_data=decoded_data, input_plot=input_plot, output_plot=output_plot, matching_percentage=percentage, comparison=comparison)

    return render_template('steganography.html')

@app.route('/cartonize', methods=['GET', 'POST'])
def cartonize():
    cartoonizer = Cartoonizer()
    if request.method == 'POST':
        # Assuming you have a form to upload an image
        uploaded_image = request.files['img']

        if uploaded_image:
            # Save the uploaded image temporarily (you may want to save it permanently)
            original_image_path = "static/uploads/ori_image.jpg"
            uploaded_image.save(original_image_path)

            # Apply the cartoon effect using the Cartoonizer class
            cartoon_image = cartoonizer.render(original_image_path)
            cartoon_image_path = "static/uploads/cartoon_image.jpg"
            cv2.imwrite(cartoon_image_path, cartoon_image)

            # Return the cartoonized image and original image path to the user or display it on the page
            # You can use a template to display the images or return them directly
            return render_template('cartoonize.html', original_image=original_image_path, cartoon_image=cartoon_image_path)

    return render_template('cartoonize.html')


@app.route('/opening', methods=['GET', 'POST'])
def opening():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binarized_img, cv2.MORPH_OPEN, kernel, iterations=1)

        opening_image_path = os.path.join(app.config['UPLOAD'], 'opening_image.jpg')
        cv2.imwrite(opening_image_path, opening)

        return render_template('opening.html', img=img_path, opening_img=opening_image_path)
    return render_template('opening.html')

@app.route('/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, kernel, iterations=4)

        closing_image_path = os.path.join(app.config['UPLOAD'], 'closing_image.jpg')
        cv2.imwrite(closing_image_path, closing)

        return render_template('closing.html', img=img_path, closing_img=closing_image_path)
    return render_template('closing.html')

@app.route('/dilasi', methods=['GET', 'POST'])
def dilasi():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        kernel = np.ones((3, 3), np.uint8) 
        
        img_dilation = cv2.dilate(img, kernel, iterations=1)

        img_erosion = cv2.erode(img, kernel, iterations=1) 


        dilasi_image_path = os.path.join(app.config['UPLOAD'], 'dilasi_image.jpg')
        erosion_image_path = os.path.join(app.config['UPLOAD'], 'erosi_image.jpg')

        cv2.imwrite(dilasi_image_path, img_dilation)
        cv2.imwrite(erosion_image_path, img_erosion)


        return render_template('dilasi.html', img=img_path, dilasi_img=dilasi_image_path, erosi_img=erosion_image_path)
    return render_template('dilasi.html')

# @app.route('/erosi', methods=['GET', 'POST'])
# def erosi():
#     if request.method == 'POST':
#         file = request.files['img']
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD'], filename))
#         img_path = os.path.join(app.config['UPLOAD'], filename)

#         img = cv2.imread(img_path)

#         kernel = np.ones((3, 3), np.uint8) 
        
#         img_erosion = cv2.erode(img, kernel, iterations=1) 

#         erosion_image_path = os.path.join(app.config['UPLOAD'], 'erosi_image.jpg')
#         cv2.imwrite(erosion_image_path, img_erosion)

#         return render_template('erosi.html', img=img_path, erosi_img=erosion_image_path)
#     return render_template('erosi.html')


@app.route('/bilinear', methods=['GET', 'POST'])
def bilinear():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  

        # scaling menggunakan bilinear interpolation
        scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        scaled_img_nearest = cv2.resize(img, None, fx=scale_x, fy=scale_y,interpolation=cv2.INTER_NEAREST)
        scaled_img_lanczos = cv2.resize(img, None, fx=scale_x, fy=scale_y,interpolation=cv2.INTER_LANCZOS4)



        # Menyimpan gambar
        scaled_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        scaled_imageN_path = os.path.join(app.config['UPLOAD'], 'scaled_imageN.jpg')
        scaled_imageL_path = os.path.join(app.config['UPLOAD'], 'scaled_imageL.jpg')
        cv2.imwrite(scaled_image_path, scaled_img)
        cv2.imwrite(scaled_imageN_path, scaled_img_nearest)
        cv2.imwrite(scaled_imageL_path, scaled_img_lanczos)


        return render_template('bilinear.html', img=img_path, img_bilinear=scaled_image_path, img_nearest=scaled_imageN_path, img_lanczos=scaled_imageL_path)
    return render_template('bilinear.html')


@app.route('/lanczos', methods=['GET', 'POST'])
def lanczos():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        # Mendefinisikan scale
        scale_x = int(float(request.form['scale_x'])) 
        scale_y = int(float(request.form['scale_y'])) 

        # scaling menggunakan lanczos interpolation
        scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y,interpolation=cv2.INTER_LANCZOS4)

        # Menyimpan gambar
        scaled_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(scaled_image_path, scaled_img)

        return render_template('lanczos.html', img=img_path, img_lanczos=scaled_image_path)
    return render_template('lanczos.html')

import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color, io
import skimage.util.noise as noise
from numpy import array

@app.route('/restoration', methods=['GET', 'POST'])
def restoration():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        image = cv2.imread(img_path)
        image = color.rgb2gray(image)

        gn=noise.random_noise(image,mode='s&p')
        result = ndimage.uniform_filter(gn, 7)
        # scaling menggunakan lanczos interpolation

        restored_lowpass_path = os.path.join(app.config['UPLOAD'], 'lowpass.png')

        #rank-older filtering
        cross=array([[0,1,0],[1,1,1],[0,1,0]])
        gn_rank=noise.random_noise(image,mode='s&p')
        result_rank = ndimage.median_filter(gn_rank, footprint=cross)


        fig, axes = plt.subplots(nrows=1, ncols=2)

        ax = axes.ravel()
        ax[0].imshow(gn, cmap='gray')
        ax[0].set_title("Salt Pepper")
        ax[1].imshow(result, cmap='gray')
        ax[1].set_title("Low pass filter")
        plt.tight_layout()
        fig.savefig(restored_lowpass_path)


        rank_image_path = os.path.join(app.config['UPLOAD'], 'rank.png')

        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax = axes.ravel()
        ax[0].imshow(gn_rank, cmap='gray')
        ax[0].set_title("Salt Pepper")
        ax[1].imshow(result_rank, cmap='gray')
        ax[1].set_title("Rank order filter")
        plt.tight_layout()
        fig.savefig(rank_image_path)

        av=array([[0,1,0],[1,1,1],[0,1,0]])/8.0
        image_sp=noise.random_noise(image,mode='s&p')
        image_sp_av=ndimage.convolve(image_sp,av)
        D=0.48
        r=(abs(image_sp-image_sp_av)>D)*1.0
        outlier_image = r*image_sp_av+(1-r)*image_sp

        outlier_image_path = os.path.join(app.config['UPLOAD'], 'outlier.png')
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax = axes.ravel()
        ax[0].imshow(gn_rank, cmap='gray')
        ax[0].set_title("Salt Pepper")
        ax[1].imshow(outlier_image, cmap='gray')
        ax[1].set_title("outlier filter")
        plt.tight_layout()
        fig.savefig(outlier_image_path)

        perbandingan_image_path = os.path.join(app.config['UPLOAD'], 'perbandingan.png')
        fig, axes = plt.subplots(nrows=1, ncols=3)
        ax = axes.ravel()
        ax[0].imshow(result, cmap='gray')
        ax[0].set_title("low pass filter")
        ax[0].axis('off')  # Turn off axis for the first subplot
        ax[1].imshow(result_rank, cmap='gray')
        ax[1].set_title("Rank order filter")
        ax[1].axis('off')
        ax[2].imshow(outlier_image, cmap='gray')
        ax[2].set_title("outlier filter")
        ax[2].axis('off')
        plt.tight_layout()
        fig.savefig(perbandingan_image_path)

        # Menyimpan gambar
        return render_template('image_restoration.html', img_lowpass = restored_lowpass_path, 
                                                        img_rank = rank_image_path,
                                                        img_outlier = outlier_image_path,
                                                        perbandingan = perbandingan_image_path)
    return render_template('image_restoration.html')

@app.route('/add_noise', methods=['GET', 'POST'])
def add_noise():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path, 
                 cv2.IMREAD_GRAYSCALE) 
        # Mendefinisikan scale
        

        # scaling menggunakan lanczos interpolation

        # Menyimpan gambar
        noised_image_path = os.path.join(app.config['UPLOAD'], 'noised_image.jpg')
        cv2.imwrite(noised_image_path, 
            add_noise(img))
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # ax = axes.ravel()
        # ax[0].imshow(img, cmap='gray')
        # ax[0].set_title("Salt Pepper")
        # ax[1].imshow(noised_image, cmap='gray')
        # ax[1].set_title("outlier filter")
        # plt.tight_layout()
        # fig.savefig(noised_image_path)

        return render_template('noised.html', img=img_path, img_noise=noised_image_path)
    return render_template('noised.html')


import random 
def add_noise(img): 
  
    # Getting the dimensions of the image 
    row , col = img.shape 
      
    # Randomly pick some pixels in the 
    # image for coloring them white 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300, 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to white 
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in 
    # the image for coloring them black 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300 , 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to black 
        img[y_coord][x_coord] = 0
          
    return img 

if __name__ == '__main__': 
    app.run(debug=True,port=8001)


