# hybrid_compression_pipeline.py

# Hybrid Medical Image Compression: JPEG + Autoencoder
# Full Pipeline: Model Design, Development, Training, Evaluation, and Output Logging

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
from math import log10
import pandas as pd
import matplotlib.pyplot as plt

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
DATASET_PATH = 'medical_images_dataset/'  # Folder with subfolders: 'MRI', 'Xray', 'CT'
RESULTS_CSV = 'compression_results.csv'

def load_images(modality_folder):
    images = []
    for filename in os.listdir(modality_folder):
        img_path = os.path.join(modality_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            images.append(img)
    return np.array(images)

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / np.sqrt(mse))

def ssim(img1, img2):
    return tf.image.ssim(tf.convert_to_tensor(img1), tf.convert_to_tensor(img2), max_val=1.0).numpy().mean()

def build_autoencoder():
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def compress_jpeg(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, encimg = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
    decimg = cv2.imdecode(encimg, 0).astype('float32') / 255.0
    return decimg

def evaluate_compression(original, compressed):
    return {
        'PSNR': psnr(original, compressed),
        'SSIM': ssim(original, compressed),
        'PixelAccuracy': np.mean((original > 0.5) == (compressed > 0.5))
    }

def plot_metrics(df):
    techniques = df['Technique'].unique()
    metrics = ['PSNR', 'SSIM', 'PixelAccuracy']

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for modality in df['Modality'].unique():
            subset = df[df['Modality'] == modality]
            plt.plot(subset['Technique'], subset[metric], marker='o', label=modality)
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xlabel('Compression Technique')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')
        plt.show()

def main():
    all_results = []
    modalities = ['Xray', 'MRI', 'CT']

    for modality in modalities:
        print(f"--- Processing {modality} images ---")
        imgs = load_images(os.path.join(DATASET_PATH, modality))
        imgs = imgs.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        jpeg_metrics = []
        for img in imgs:
            jpeg_img = compress_jpeg(img.squeeze())
            metrics = evaluate_compression(img.squeeze(), jpeg_img)
            jpeg_metrics.append(metrics)

        autoencoder = build_autoencoder()
        autoencoder.fit(imgs, imgs, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        autoencoded_imgs = autoencoder.predict(imgs)
        ae_metrics = [evaluate_compression(o.squeeze(), r.squeeze()) for o, r in zip(imgs, autoencoded_imgs)]

        hybrid_imgs = []
        for img in imgs:
            jpeg_img = compress_jpeg(img.squeeze())
            jpeg_img = jpeg_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            hybrid_img = autoencoder.predict(jpeg_img)
            hybrid_imgs.append(hybrid_img.squeeze())
        hybrid_metrics = [evaluate_compression(o.squeeze(), r) for o, r in zip(imgs, hybrid_imgs)]

        for idx, label in enumerate(['JPEG', 'Autoencoder', 'Hybrid']):
            metrics_set = [jpeg_metrics, ae_metrics, hybrid_metrics][idx]
            avg_psnr = np.mean([m['PSNR'] for m in metrics_set])
            avg_ssim = np.mean([m['SSIM'] for m in metrics_set])
            avg_acc = np.mean([m['PixelAccuracy'] for m in metrics_set])
            all_results.append({
                'Modality': modality,
                'Technique': label,
                'PSNR': avg_psnr,
                'SSIM': avg_ssim,
                'PixelAccuracy': avg_acc
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print("Compression results saved to:", RESULTS_CSV)
    plot_metrics(results_df)

if __name__ == "__main__":
    main()
