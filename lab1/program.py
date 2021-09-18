import numpy as np
import cv2
import matplotlib
from typing import Tuple,Any,List
matplotlib.use('Agg')
import time

EPS = 0.9
BBOX1 = np.array([150,100,200,400])
BBOX2 = np.array([270,100,320,400])

def gauss_prob(
    x : np.ndarray, 
    mean : Tuple[np.ndarray], 
    denominator : Tuple[float], 
    cov_inv :Tuple[np.ndarray]
    ) -> float:
    x_m = x - mean
    numerator = np.exp(-(np.dot(np.dot(x_m.T,cov_inv),x_m)) / 2)
    return (numerator/denominator)/100


def get_neighbors(matrix : np.ndarray, i : int, j : int):
    neighbors = []
    if i > 0:
        neighbors.append(matrix[i-1, j])
    if j > 0:
        neighbors.append(matrix[i, j-1])
    if j < matrix.shape[1] - 1:
        neighbors.append(matrix[i, j+1])
    if i < matrix.shape[0] - 1:
        neighbors.append(matrix[i+1, j])
    return neighbors

def compute_obervation(
    label_matrix : np.ndarray, 
    image : np.ndarray,
    mean : Tuple[np.ndarray], 
    denominator : Tuple[float],
    cov_inv :  Tuple[np.ndarray]
    ) -> np.ndarray:
    new_label_matrix = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            neighbors = get_neighbors(label_matrix,i,j)
            pixels = image[i, j] + EPS * np.linalg.norm(neighbors[:4])
            p1 = gauss_prob(pixels, mean[0], denominator[0], cov_inv[0])
            p2 = gauss_prob(pixels, mean[1], denominator[1], cov_inv[1])
            if p1>=p2:
                new_label_matrix[i,j] = 0
            else:
                new_label_matrix[i,j] = 1
    return new_label_matrix

def get_image_params(
    image : np.ndarray,
    bbox1:np.ndarray,
    bbox2:np.ndarray
    ) -> Tuple[Tuple[np.ndarray],Tuple[np.ndarray],Tuple[np.ndarray]]:
    area1 = image[bbox1[0]:bbox1[2],bbox1[1]:bbox1[3]]
    area2 = image[bbox2[0]:bbox2[2],bbox2[1]:bbox2[3]]
    m1 = np.mean(area1, axis = (0,1))
    m2 = np.mean(area2, axis = (0,1))
    s1 = get_cov_matrix(area1)
    s2 = get_cov_matrix(area2)
    cov_inverse1 = np.linalg.inv(s1)
    cov_inverse2 = np.linalg.inv(s2) 
    cov_det1 = np.linalg.det(s1)
    cov_det2 = np.linalg.det(s2)
    return (m1,m2),(cov_inverse1,cov_inverse2),(cov_det1,cov_det2)

def get_cov_matrix(image : np.ndarray) -> np.ndarray:
    image = image.reshape(-1, 3)
    meanX = np.mean(image, axis = 0)
    lenX = image.shape[0]
    X = image - meanX
    cov_matrix = X.T.dot(X)/lenX
    return cov_matrix

def get_init_prob(
    image : np.ndarray, 
    mean : Tuple[np.ndarray], 
    denominator : Tuple[float],
    cov_inv :  Tuple[np.ndarray]
    ) -> np.ndarray:
    label_matrix = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            p1 = gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0])
            p2 = gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1])
            if p1>=p2:
                label_matrix[i,j] = 0
            else:
                label_matrix[i,j] = 1
    return label_matrix

def gibbs_sampler(image : np.ndarray,iterations: int = 100) -> Tuple[np.ndarray,List[np.ndarray]]:
    mean,cov_inv,cov_det = get_image_params(image,BBOX1,BBOX2)
    image = cv2.resize(image,(256,256))
    denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
    label_matrix = get_init_prob(image, mean, denominator, cov_inv)
    results = []
    for iteration in range(iterations):
        print(iteration)
        label_matrix = compute_obervation(label_matrix, image, mean, denominator, cov_inv)
        results.append(label_matrix)
    
    return label_matrix,results


if __name__ == '__main__':
    image_path = 'image.jpg'
    image = cv2.imread(image_path)
    start = time.time()
    seg_image, results = gibbs_sampler(image)
    end = time.time()
    print("TIME = ",end-start)
    for i,mask in enumerate(results):
        stacked_img = np.stack((mask,)*3, axis=-1)*255
        cv2.imshow('Window', stacked_img)
        key = cv2.waitKey(1000)
        if key == 27:
            cv2.destroyAllWindows()
            break