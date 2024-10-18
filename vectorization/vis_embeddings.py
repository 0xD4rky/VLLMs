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

## --------------------------- PLOTTING 3D POINTS --------------------------##


"""
With the statistics in hand now, focus now will be on plotting 2048 dim vectors into 3D.
"""


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')


x, y, z = np.indices(embedding_3d.shape)


sc = ax.scatter(x, y, z, c=embedding_3d.flatten(), cmap='plasma', s=5)


ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')
ax.set_title('3D Visualization of Single Image Embedding', color='white')


ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')


cbar = plt.colorbar(sc)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')


ax.view_init(elev=20, azim=45)


plt.tight_layout()
plt.show()


print(f"Embedding shape: {embedding.shape}")
print(f"Embedding mean: {np.mean(embedding)}")
print(f"Embedding std: {np.std(embedding)}")
print(f"Embedding min: {np.min(embedding)}")
print(f"Embedding max: {np.max(embedding)}")