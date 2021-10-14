import numpy as np
import cv2
import matplotlib
from typing import Tuple,Any,List
matplotlib.use('Agg')
import time

EPS = 0.3
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
            factor = 1
            for neighbor in neighbors:
                if factor == label_matrix[i,j]:
                    factor*=EPS
                else: factor*=(1-EPS)
            p1 = gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0]) * factor
            p2 = gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1]) * factor
            if p1>=p2:
                new_label_matrix[i,j] = 0
            else:
                new_label_matrix[i,j] = 1
    return new_label_matrix

def get_image_params(
    area1 : np.ndarray,
    area2 : np.ndarray,
    ) -> Tuple[Tuple[np.ndarray],Tuple[np.ndarray],Tuple[np.ndarray]]:
    m1 = np.mean(area1, axis = 0)
    m2 = np.mean(area2, axis = 0)
    s1 = np.cov(area1,rowvar = False)
    s2 = np.cov(area2,rowvar = False)
    cov_inverse1 = np.linalg.inv(s1)
    cov_inverse2 = np.linalg.inv(s2) 
    cov_det1 = np.linalg.det(s1)
    cov_det2 = np.linalg.det(s2)
    return (m1,m2),(cov_inverse1,cov_inverse2),(cov_det1,cov_det2)


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
    area1 = image[BBOX1[0]:BBOX1[2],BBOX1[1]:BBOX1[3]]
    area1 = area1.reshape(area1.shape[0]*area1.shape[1],3)
    area2 = image[BBOX2[0]:BBOX2[2],BBOX2[1]:BBOX2[3]]
    area2 = area2.reshape(area2.shape[0]*area2.shape[1],3)
    mean,cov_inv,cov_det = get_image_params(area1,area2)
    image = cv2.resize(image,(256,256))
    denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
    label_matrix = get_init_prob(image, mean, denominator, cov_inv)
    results = []
    for iteration in range(iterations):
        print(iteration)
        label_matrix = compute_obervation(label_matrix, image, mean, denominator, cov_inv)
        area1 = image[label_matrix==0]
        area2 = image[label_matrix==1]
        mean,cov_inv,cov_det = get_image_params(area1,area2)
        denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
        results.append(label_matrix)
    return label_matrix,results


if __name__ == '__main__':
    image_path = 'lab1/image.jpg'
    image = cv2.imread(image_path)
    start = time.time()
    seg_image, results = gibbs_sampler(image)
    stacked_img = np.stack((seg_image,)*3, axis=-1)
    stacked_img[:,:,0]*=255
    cv2.imwrite("lab1/result.jpg",stacked_img)
    end = time.time()
    print("TIME = ",end-start)
    for i,mask in enumerate(results):
        stacked_img = np.stack((mask,)*3, axis=-1)*255
        cv2.imwrite(f'lab1/results/image_{i}.jpg',stacked_img)