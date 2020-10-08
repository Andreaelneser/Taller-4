import numpy as np
import cv2

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


def recreate_image (centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

if __name__ == '__main__':


    print("Seleccione el método con el que desea trabajar: kmeans o gmm ") #Se pregunta al usuario cuál metodo desea
    metodo = input()
    print("Su método escogido es", metodo)
    f_dist = np.zeros(10, float)

    for n_cl in range(10):
        print("Se está haciendo clustering para :", n_cl + 1, 'cluster(s)' )
        n_colors = n_cl + 1  # Se colocan los valores de centro de color

        image = cv2.imread("bandera.png")    #Se carga la imagen
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #se convierte la imagen a RGB

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        image = np.array(image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        if metodo == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample)
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        print("Predicting color indices on the full image (GMM)")
        t0 = time()
        if metodo == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
        else:
            print("Predicting color indices on the full image (Kmeans)")
            labels = model.predict(image_array)
            centers = model.cluster_centers_
        #    print("El label es", labels)
        #    print("El centro es", centers)

        print("done in %0.3fs." % (time() - t0))

        dist = np.zeros(n_colors, float)
        for lb in range(labels.shape[0]):
            cl = labels[lb]
            dist[cl] += np.sqrt(((image_array[lb, 0] - centers[cl, 0]) ** 2) + ((image_array[lb, 1] - centers[cl, 1])**2) + ((image_array[lb, 2] - centers[cl, 2]) ** 2))
        f_dist[n_cl] = np.sum(dist)
        print("El valor de la distancia intra cluster es", f_dist[n_cl])

        plt.figure(n_cl + 1)
        plt.clf()
        plt.axis("Off")
        plt.title("Quantidez image ({} color, method = {}" .format(n_colors, metodo))
        plt.imshow(recreate_image(centers, labels, rows, cols))



    # Display all results, alongside original image
    plt.figure(0)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)

    plt.figure(11)
    plt.plot(range(1, 11), f_dist, 'g', ms = 7)
    plt.title("Distancia")
    plt.xlabel("Número de clusters")
    plt.xlabel("Distancia intra cluster")
    plt.grid()
    plt.show()

