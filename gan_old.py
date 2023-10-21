import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Generate random circles with only (x, y) coordinates
def generate_random_circles(batch_size, num_circles=5):
    circles = np.random.rand(batch_size, num_circles, 2)  # Random (x, y) coordinates
    return circles

# Create the generator model
def build_generator(latent_dim, num_circles):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dense(256),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dense(2 * num_circles),  # Output format: (x, y) coordinates for each circle
        Reshape((num_circles, 2))
    ])
    return model

# Create the discriminator model
def build_discriminator(num_circles):
    model = Sequential([
        Flatten(input_shape=(num_circles, 2)),  # Flattens (x, y) coordinates
        Dense(512),
        LeakyReLU(alpha=0.01),
        Dense(256),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')  # Output: Real or Fake (1 or 0)
    ])
    return model

# Combine the generator and discriminator into a GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_circles = generator(gan_input)
    gan_output = discriminator(generated_circles)
    gan = Model(gan_input, gan_output)
    return gan

# GAN training function
def train_gan(generator, discriminator, gan, epochs=1000, batch_size=128):
    for epoch in tqdm(range(epochs)):
        # Generate real and fake data
        real_circles = generate_random_circles(batch_size)
        noise = np.random.randn(batch_size, latent_dim)
        generated_circles = generator.predict(noise)

        # Combine real and fake data
        x_combined = np.concatenate([real_circles, generated_circles])

        # Labels for the real and fake data
        y_combined = np.zeros(2 * batch_size)
        y_combined[:batch_size] = 0.9  # Add some label smoothing for real data

        # Train the discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x_combined, y_combined)

        # Train the generator (through the GAN model)
        noise = np.random.randn(batch_size, latent_dim)
        y_fake = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_fake)

        # Rest of the code remains unchanged


        # Print the progress
        # Visualization: Plot generated circles
        if epoch % 200 == 0:
            plt.figure(figsize=(8, 8))

            # Scaling the generated circles' coordinates to match the range of real circle coordinates
            generated_circles_scaled = generated_circles * 0.5 + 0.5

            # Plot real circles
            plt.scatter(real_circles[0][:, 0], real_circles[0][:, 1], color='red', label='Real Circles')

            # Plot generated circles
            plt.scatter(generated_circles_scaled[0][:, 0], generated_circles_scaled[0][:, 1], color='blue', label='Generated Circles')

            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title(f"Generated Circles at Epoch {epoch}")
            plt.savefig(f"epoch_{epoch}.png")
            plt.close()

# Set parameters
latent_dim = 100  # Dimension of the generator's input noise vector
num_circles = 5   # Number of circles to generate in each image

# Build models
generator = build_generator(latent_dim, num_circles)
discriminator = build_discriminator(num_circles)
gan = build_gan(generator, discriminator)

# Compile models
generator.compile(loss='binary_crossentropy', optimizer=Adam())
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Train the GAN
train_gan(generator, discriminator, gan)