import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    image1 = cv2.imread("lena.png")  # Se carga la imagen
    image2 = cv2.imread("lena_warped.png")  # Se carga la imagen
    image1= cv2.cvtColor(image1, (cv2.COLOR_BGR2RGB))
    image2= cv2.cvtColor(image2, (cv2.COLOR_BGR2RGB))

    # show image1
    fig, ax = plt.subplots()
    ax.imshow(image1)

    # select points 1
    cor1 = plt.ginput(0, 0)
    print("Coordenadas imagen1:\n", cor1)
    print("")

    # show image 2
    fig, ax = plt.subplots()
    ax.imshow(image2)

    # select points 2
    cor2 = plt.ginput(0, 0)
    print("Coordenadas imagen2:\n", cor2)

    # affine
    pts1 = np.float32([cor1])
    pts2 = np.float32([cor2])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    image_affine = cv2.warpAffine(image2, M_affine, image2.shape[:2])
    image_affine= cv2.cvtColor(image_affine, (cv2.COLOR_BGR2RGB))


    cv2.imshow("Transformacion afin", image_affine)
    cv2.waitKey(0)

    print("La matriz afín es:\n", M_affine)

    #scaling
    sx = np.sqrt((M_affine[0, 0] + M_affine[1, 0])**2)
    sy = np.sqrt((M_affine[0, 1] + M_affine[1, 1])**2)
    M_s = np.float32([[sx, 0, 0], [0, sy, 0]])
    image_scale = cv2.warpAffine(image_affine, M_s, (700, 700))

    # rotation
    theta = -np.arctan(M_affine[1, 0]/M_affine[0, 0])
    theta_rad = theta * np.pi / 180
    # M_rot = np.float32([[np.cos(theta_rad), -np.sin(theta_rad), 0],
    #                    [np.sin(theta_rad), np.cos(theta_rad), 0]])
    cx = image_affine.shape[1] / 2
    cy = image_affine.shape[0] / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), theta, 1)
    image_rotation = cv2.warpAffine(image_affine, M_rot, image_affine.shape[:2])

    # translation
    tx = ((M_affine[0,2]*np.cos(theta))-(M_affine[1,2]*np.sin(theta)))/sx
    ty = ((M_affine[0,2]*np.sin(theta))-(M_affine[1,2]*np.cos(theta)))/sy
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    image_translation = cv2.warpAffine(image_affine, M_t, (image_affine.shape[1], image_affine.shape[0]))

    # similarity
    M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                        [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
    image_similarity = cv2.warpAffine(image1, M_sim, image1.shape[:2])
    image_similarity= cv2.cvtColor(image_similarity, (cv2.COLOR_BGR2RGB))
    cv2.imshow("Transformacion imagen de similitud", image_similarity)
    cv2.waitKey(0)

    #norma de imagen I2
    norma = abs(np.linalg.norm(np.subtract(cor1, cor2), ord=1))
    print("La norma de error es:\n" ,norma)
