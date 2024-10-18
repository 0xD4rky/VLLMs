import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddings import *

embedding_np = embedding.cpu().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(embedding_np[:100])
plt.title("First 100 values of the embedding")
plt.xlabel("Index")
plt.ylabel("Value")

plt.subplot(1, 3, 2)
plt.hist(embedding_np, bins=50)
plt.title("Histogram of embedding values")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.boxplot(embedding_np)
plt.title("Box plot of embedding values")
plt.ylabel("Value")

plt.tight_layout()
plt.show()

print(f"Embedding mean: {np.mean(embedding_np):.4f}")
print(f"Embedding std: {np.std(embedding_np):.4f}")
print(f"Embedding min: {np.min(embedding_np):.4f}")
print(f"Embedding max: {np.max(embedding_np):.4f}")
print(f"Embedding median: {np.median(embedding_np):.4f}")


cube_size = int(np.cbrt(embedding.shape[0]))
embedding_3d = embedding[:cube_size**3].reshape((cube_size, cube_size, cube_size))

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the cube
x, y, z = np.indices(embedding_3d.shape)

# Plot the cube
sc = ax.scatter(x, y, z, c=embedding_3d.flatten(), cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Single Image Embedding')

# Add a color bar
plt.colorbar(sc)

# Show plot
plt.tight_layout()
plt.show()

# Print some statistics about the embedding
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding mean: {np.mean(embedding)}")
print(f"Embedding std: {np.std(embedding)}")
print(f"Embedding min: {np.min(embedding)}")
print(f"Embedding max: {np.max(embedding)}")