import numpy as np
import cv2
import matplotlib
from typing import Tuple,Any,List
matplotlib.use('Agg')
import time
from scipy.stats import multivariate_normal as mv_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

EPS = 0.1

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
                if neighbor == label_matrix[i,j]:
                    factor*=EPS
                else: factor*=(1-EPS)
            p1 = gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0]) * factor
            p2 = gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1]) * factor
            p1 = p1/(p1+p2)
            p2 = p2/(p1+p2)
            if p1>=p2:
                new_label_matrix[i,j] = 0
            else:
                new_label_matrix[i,j] = 1
    return new_label_matrix

def EM_algo(img):
    p_0 = 0.5
    p_1 = 0.5
    mean_0 = np.random.randint(1,255,size=3)
    mean_1 = np.random.randint(1,255,size=3)
    cov_0 = np.random.randint(-1000,1000,size=(3,3))
    cov_0 = cov_0.T @ cov_0
    cov_1 = np.random.randint(-1000,1000,size=(3,3))
    cov_1 = cov_1.T @ cov_1
    for _ in tqdm(range(100)):
        
        dens_arr0 = mv_normal.pdf(img, mean_0, cov_0)
        dens_arr1 = mv_normal.pdf(img, mean_1, cov_1)
        
        #2a
        alphas_0 = p_0*dens_arr0
        alphas_1 = p_1*dens_arr1
        sum_alphas = alphas_0 + alphas_1
        alphas_0 = alphas_0/sum_alphas
        alphas_1 = alphas_1/sum_alphas
        
        #2b 
        p_0 = alphas_0.mean()
        p_1 = alphas_1.mean()
        
        #2c
        alphas_0_temp = np.zeros(img.shape)
        alphas_1_temp = np.zeros(img.shape)
        for s in [0,1,2]:
            alphas_0_temp[:,:,s] = alphas_0
            alphas_1_temp[:,:,s] = alphas_1
        mean_0 = ( (alphas_0_temp * img).sum( axis=(0,1) ))/(alphas_0.sum())
        mean_1 = ( (alphas_1_temp * img).sum( axis=(0,1) ))/(alphas_1.sum())
        
        alphas_0_temp = np.sqrt(alphas_0_temp)
        numerator0 = ((alphas_0_temp)*(img - mean_0)).reshape(3,-1)
        numerator0 = numerator0 @ numerator0.T
        cov_0 = numerator0/(alphas_0.sum())
        alphas_1_temp = np.sqrt(alphas_1_temp)
        numerator1 = ((alphas_1_temp)*(img - mean_1)).reshape(3,-1)
        numerator1 = numerator1 @ numerator1.T
        cov_1 = numerator1/(alphas_1.sum())

    return mean_0.astype(int),mean_1.astype(int),cov_0.astype(int),cov_1.astype(int)

def get_image_params(
    image,
    ) -> Tuple[Tuple[np.ndarray],Tuple[np.ndarray],Tuple[np.ndarray]]:
    m1,m2,cov1,cov2 = EM_algo(image)
    cov_inverse1 = np.linalg.inv(cov1)
    cov_inverse2 = np.linalg.inv(cov2) 
    cov_det1 = np.linalg.det(cov1)
    cov_det2 = np.linalg.det(cov2)
    
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

def gibbs_sampler(image : np.ndarray,iterations: int = 20) -> Tuple[np.ndarray,List[np.ndarray]]:
    mean,cov_inv,cov_det = get_image_params(image)
    denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
    label_matrix = get_init_prob(image, mean, denominator, cov_inv)
    results = []
    for _ in tqdm(range(iterations)):
        label_matrix = compute_obervation(label_matrix, image, mean, denominator, cov_inv)
        results.append(label_matrix)
    return label_matrix,results

if __name__ == '__main__':
    image_path = 'lab3/image.jpg'
    image = plt.imread(image_path)
    image = cv2.resize(image,(256,256))
    start = time.time()
    seg_image, results = gibbs_sampler(image)
    stacked_img = np.stack((seg_image,)*3, axis=-1)
    stacked_img[:,:,0]*=255
    cv2.imwrite("lab3/result.jpg",stacked_img)
    end = time.time()
    print("TIME = ",end-start)
    for i,mask in enumerate(results):
        stacked_img = np.stack((mask,)*3, axis=-1)*255
        cv2.imwrite(f'lab3/results/image_{i}.jpg',stacked_img)