import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs('static/uploads', exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def display_images(images, titles, save_path='static/uploads/result.png'):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()

def process_image(image_path):
    image = cv2.imread(image_path)
    
    # Denoising the image
    gaussian_noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    denoised_mean = cv2.blur(noisy_image, (5, 5))
    denoised_median = cv2.medianBlur(noisy_image, 3)    
    
    # Sharpening the image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    
    # Edge Detection
    sobel_x = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge = cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)
    
    prewitt_x = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitt_y = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    prewitt_edge = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float)).astype(np.uint8)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edge = cv2.Canny(gray_image, 100, 200)
    
    sobel_edge_rgb = cv2.merge([sobel_edge, sobel_edge, sobel_edge])
    prewitt_edge_rgb = cv2.merge([prewitt_edge, prewitt_edge, prewitt_edge])
    canny_edge_rgb = cv2.merge([canny_edge, canny_edge, canny_edge])
    
    images = [
        noisy_image, denoised_mean, denoised_median,
        sharpened_image, sobel_edge_rgb, prewitt_edge_rgb, canny_edge_rgb
    ]
    titles = [
        'Noisy (Gaussian)', 'Denoised (Mean)', 'Denoised (Median)',
        'Sharpened', 'Sobel Edge', 'Prewitt Edge', 'Canny Edge'
    ]
    
    display_images(images, titles)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            process_image(image_path)
            return redirect(url_for('display_result', filename='result.png'))
    return render_template('upload.html')

@app.route('/result/<filename>')
def display_result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)