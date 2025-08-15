from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def image_to_palette_repr(imagem_path, n_cores=5, reduce_factor=1):
    """
    Dado uma imagem em `imagem_path`, gera uma paleta de `n_cores` cores, e uma representação
    da mesma com os índices da paleta, com tamanho reduzido pelo fator `reduce_factor`.   
    """
    image = Image.open(imagem_path)
    image = image.convert('RGB')
    image_array_original = np.array(image)
    
    # Garante que as dimensões da imagem são múltiplos
    # do fator de redução
    h, w, c = image_array_original.shape
    h_reduce, w_reduce = h // reduce_factor, w // reduce_factor
    image_array = image_array_original[:h_reduce*reduce_factor, :w_reduce*reduce_factor]

    # Reduz a imagem
    image_array = image_array.reshape(h_reduce, reduce_factor, w_reduce, reduce_factor, c).mean(axis=(1, 3)).astype(np.uint8)

    pixels = image_array.reshape(-1, 3)

    # Gera a paleta de cores
    kmeans = KMeans(n_clusters=n_cores, random_state=42)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)

    # Mostrar paleta
    # plt.figure(figsize=(8, 2))
    # for i, cor in enumerate(cores):
    #     plt.subplot(1, n_cores, i+1)
    #     plt.imshow([[cor]])
    #     plt.axis('off')
    #     plt.title(str(tuple(cor)))
    # plt.show()

    # Representação da imagem como indices da paleta
    p_dist_sqr = ((pixels[:, None, :] - palette)**2).sum(axis=2)
    palette_ids = np.argmin(p_dist_sqr, axis=1)
    image_palette_repr = palette[palette_ids]

    # Imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(image_array_original)
    
    # Imagem na representação da paleta
    plt.subplot(1, 2, 2)
    plt.imshow(image_palette_repr.reshape(image_array.shape))
    
    print("paleta:\n", palette)
    print("Imagem:\n", palette_ids.reshape(image_array.shape[:2]))

    plt.show()
    