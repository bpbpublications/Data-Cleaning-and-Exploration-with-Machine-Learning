import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np

# Load a grayscale image
image = plt.imread("example_image.png")[:, :, 0]

# Perform SVD
U, S, Vt = svd(image)

# Compress image by retaining top singular values
k = 50
compressed_image = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# Generate noisy audio-like signal
signal = np.sin(np.linspace(0, 2 * np.pi, 100))
noisy_signal = signal + 0.2 * np.random.normal(size=100)

# Perform SVD-based denoising
U, S, Vt = svd(noisy_signal.reshape(-1, 1))
denoised_signal = (U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]).flatten()

# Display compressed image
plt.imshow(compressed_image, cmap='gray')
plt.title("Compressed Image with Top 50 Singular Values")
plt.axis('off')
plt.show()


# Plot denoised signal
plt.plot(noisy_signal, label='Noisy Signal')
plt.plot(denoised_signal, label='Denoised Signal', linestyle='--')
plt.legend()
plt.show()
