from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import io
import os
import base64

import mnist_loader
from network import Network

app = Flask(__name__)
net = Network([784, 30, 10])


@app.route('/')
def index():
    if not os.path.isfile('network.pkl'):
        return render_template('learn.html')

    return render_template('index.html')


@app.route('/learn')
def learn():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.SGD(list(training_data), 30, 10, 3, list(test_data))
    net.save()
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['data']
    image_encoded = data.split(",")[1]
    image_bytes = base64.urlsafe_b64decode(image_encoded)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white_bg, image).convert('RGB')

    image.save('image.png')

    # Convert to grayscale
    image_bw = image.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.Resampling.LANCZOS)

    sharpening_filter = ImageFilter.UnsharpMask()
    image_bw_resized = image_bw_resized.filter(sharpening_filter)
    image_bw_inverted = ImageOps.invert(image_bw_resized)

    image_bw_inverted.save('image-for-model.png')

    image_array = np.array(image_bw_inverted)
    image_array_normalized = image_array / 255.0
    image_array_reshaped = image_array_normalized.reshape(784, 1)

    net.load()
    result = net.feedforward(image_array_reshaped)

    prediction = np.argmax(result)
    return jsonify({"prediction": int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
