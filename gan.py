import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import skimage.metrics as skm
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import os
import tensorflow.keras.applications.inception_v3 as inception_v3

# Generator model
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(28*28*1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Discriminator model
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN model combining generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    return model

# Loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Training step for generator
def train_generator(generator, discriminator, batch_size):
    noise = tf.random.normal((batch_size, 100))
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise)
        fake_output = discriminator(generated_images)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return gen_loss

# Training step for discriminator
def train_discriminator(real_images, generator, discriminator, batch_size):
    noise = tf.random.normal((batch_size, 100))
    generated_images = generator(noise)

    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_images)
        fake_output = discriminator(generated_images)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return disc_loss

# Main training loop
def train_gan(generator, discriminator, epochs, batch_size):
    (train_images, _), (_, _) = datasets.mnist.load_data()
    train_images = (train_images - 127.5) / 127.5
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        for batch in dataset:
            d_loss = train_discriminator(batch, generator, discriminator, batch_size)
            g_loss = train_generator(generator, discriminator, batch_size)
        print(f"Epoch {epoch + 1}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")

        # Save generated images during training
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, epoch + 1)

    # Save the final generator model
    generator.save('generator_model.h5')

# Function to save generated images during training
def save_generated_images(generator, epoch, num_samples=100):
    noise = tf.random.normal((num_samples, 100))
    generated_images = generator(noise).numpy()

    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(10, 10, i + 1)
        plt.imshow((generated_images[i] + 1) / 2, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'generated_images_epoch_{epoch:03d}.png')
    plt.close()

# Function to load generator model
def load_generator_model():
    return tf.keras.models.load_model('generator_model.h5')

# Function to calculate FID
def calculate_fid(real_images, generated_images, batch_size=256, use_multiprocessing=True):
    inception_model = inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_images = ((real_images + 1.0) * 127.5).astype(np.uint8)
    generated_images = ((generated_images + 1.0) * 127.5).astype(np.uint8)

    def calculate_activation(images, model, preprocess_input):
        images = tf.image.resize(images, (299, 299))
        images = tf.tile(images, [1, 1, 1, 3])  # Convert grayscale to RGB
        images = preprocess_input(images)
        activations = model.predict(images, batch_size=batch_size, use_multiprocessing=use_multiprocessing)
        return activations

    real_activations = calculate_activation(real_images, inception_model, inception_v3.preprocess_input)
    generated_activations = calculate_activation(generated_images, inception_model, inception_v3.preprocess_input)

    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid_score

# Function to preprocess images for SSIM calculation
def preprocess_for_ssim(images):
    return (images + 1.0) / 2.0


# Function to calculate SSIM and PSNR
def calculate_ssim_psnr(real_images, generated_images):
    real_images = preprocess_for_ssim(real_images)
    generated_images = preprocess_for_ssim(generated_images)

    # Reshape images to 2D (batch_size, height * width * channels)
    real_images = real_images.reshape(real_images.shape[0], -1)
    generated_images = generated_images.reshape(generated_images.shape[0], -1)

    # Calculate SSIM and PSNR
    ssim_score = np.mean([skm.structural_similarity(real_images[i], generated_images[i], data_range=1.0, multichannel=False, win_size=7) for i in range(len(real_images))])
    psnr_score = np.mean([skm.peak_signal_noise_ratio(real_images[i], generated_images[i]) for i in range(len(real_images))])

    return ssim_score, psnr_score

# Training parameters
BATCH_SIZE = 256
EPOCHS = 2

if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    train_gan(generator, discriminator, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Load the final generator model
    generator = load_generator_model()

    # Generate some images
    noise = tf.random.normal((16, 100))
    generated_images = generator(noise).numpy()

    # Display generated images
    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((generated_images[i] + 1) / 2, cmap='gray')
        plt.axis('off')
    plt.show()

    # Load real images (you should replace this with your own dataset or real images)
    (train_images, _), (_, _) = datasets.mnist.load_data()
    real_images = (train_images[:16] - 127.5) / 127.5
    real_images = real_images.reshape(real_images.shape[0], 28, 28, 1).astype('float32')

    # Evaluate FID
    fid_score = calculate_fid(real_images, generated_images)
    print("Frechet Inception Distance (FID):", fid_score)

    # Calculate SSIM and PSNR
    ssim_score, psnr_score = calculate_ssim_psnr(real_images, generated_images)
    print("Structural Similarity Index (SSIM):", ssim_score)
    print("Peak Signal-to-Noise Ratio (PSNR):", psnr_score)

    # Plot FID, SSIM, and PSNR scores
    scores = [fid_score, ssim_score, psnr_score]
    metrics = ['FID', 'SSIM', 'PSNR']

    plt.bar(metrics, scores)
    plt.ylabel('Score')
    plt.title('Comparison of FID, SSIM, and PSNR')
    plt.show()
