import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def countn_byte(image):
    image = cv2.imread(image)
    n_bytes = image.shape[0] * image.shape[1] * 3 //8
    return n_bytes

def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes):
        return ''.join([ format(i, "08b") for i in data ])
    elif isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")
    
def encode(image_name, secret_data):
    # read the image
    image = cv2.imread(image_name)
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print("[*] Maximum bytes to encode:", n_bytes)
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")
    print("[*] Encoding data...")
    # add stopping criteria
    secret_data += "====="
    data_index = 0
    # convert data to binary
    binary_secret_data = to_bin(secret_data)
    # size of data to hide
    data_len = len(binary_secret_data)
    for row in image:
        for pixel in row:
            # convert RGB values to binary format
            r, g, b = to_bin(pixel)
            # modify the least significant bit only if there is still data to store
            if data_index < data_len:
                # least significant red pixel bit
                pixel[0] = int(r[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant green pixel bit
                pixel[1] = int(g[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant blue pixel bit
                pixel[2] = int(b[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            # if data is encoded, just break out of the loop
            if data_index >= data_len:
                break
    return image

def decode(image_name):
    print("[+] Decoding...")
    # read the image
    image = cv2.imread(image_name)
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    # split by 8-bits
    all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    return decoded_data[:-5]

def histogram(input_image, output_image):
    input = cv2.imread(input_image)
    output = cv2.imread(output_image)

    # Get the dimensions (size) of both images
    # Get the dimensions (size) of both images
    input_size = input.shape[:2]  # [:2] gives you (height, width)
    output_size = output.shape[:2]

    # Display the images and their sizes
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
    plt.title(f'Input Image (Size: {input_size[1]}x{input_size[0]})')

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f'Output Image (Size: {output_size[1]}x{output_size[0]})')

    plt.savefig("static/uploads/image_comparison.png")


    chans_input = cv2.split(input)
    chans_output = cv2.split(output)

    colors = ("b", "g", "r")



    plt.figure(figsize=(16, 8))
    plt.title("'Flattened' Color Histogram input")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color1) in zip(chans_input, colors):
    # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color1)
        plt.xlim([0, 256])
    
    plt.savefig("static/uploads/input_histogram.png")

    plt.clf()

    plt.title("'Flattened' Color Histogram output")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")


    for (chan, color1) in zip(chans_output, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color1)
        plt.xlim([0, 256])

    plt.savefig("static/uploads/output_histogram.png")
    plt.clf()

        # Check if the dimensions of the images are the same
    if input.shape != output.shape:
        print("Images have different dimensions. Cannot compare bit-by-bit.")
    else:
        # Convert images to binary format
        input_binary = np.unpackbits(input)
        output_binary = np.unpackbits(output)

        # Perform bit-wise comparison
        bit_comparison = np.equal(input_binary, output_binary)

        # Calculate the percentage of matching bits
        matching_percentage = np.mean(bit_comparison) * 100

        # print(f"Percentage of matching bits: {matching_percentage:.2f}%")


    return "static/uploads/input_histogram.png", "static/uploads/output_histogram.png", matching_percentage, "static/uploads/image_comparison.png"

