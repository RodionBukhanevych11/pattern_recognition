import numpy as np
import cv2
import matplotlib
from typing import Tuple,Any,List
import time
from tqdm import tqdm


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
    image : np.ndarray,
    mean : Tuple[np.ndarray], 
    denominator : Tuple[float],
    cov_inv :  Tuple[np.ndarray],
    eps : float
    ) -> np.ndarray:
    new_label_matrix = np.zeros((image.shape[0],image.shape[1]))
    for i in tqdm(range(image.shape[0]-1)):
        for j in range(image.shape[1]-1):
            neighbors = get_neighbors(image,i,j)
            neighbors_weights = []
            for neighbor in neighbors:
                p0_0 = np.log(gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0])) + np.log(1-eps)
                p0_1 = np.log(gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0])) + np.log(eps)
                p1_0 = np.log(gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1])) + np.log(eps)
                p1_1 = np.log(gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1])) + np.log(1-eps)
                neighbors_weights.append([p0_0,p0_1,p1_0,p1_1])
            neighbors_weights = np.array(neighbors_weights)
            for i in range(15):
                overweights = []
                max_edges_0 = neighbors_weights[:,:2].max(axis = 1)
                max_edges_1 = neighbors_weights[:,2:].max(axis = 1)
                for i,neighbor_edges in enumerate(neighbors_weights):
                    overweighted_neighbor = []
                    for j,edge in enumerate(neighbor_edges):
                        if j == 0 or j==1:
                            new_edge = abs(edge - max_edges_0[i] + (max_edges_0.sum()/neighbors_weights.shape[0]))
                        elif j == 2 or j==3:
                            new_edge = abs(edge - max_edges_1[i] + (max_edges_1.sum()/neighbors_weights.shape[0]))
                        overweighted_neighbor.append(new_edge)
                    overweights.append(overweighted_neighbor)
                neighbors_weights = np.array(overweights)
            filtered_edges = []
            maximum_edge = neighbors_weights.max()
            filtered_edges = np.zeros((len(neighbors),4))
            for i,neighbor_edges in enumerate(neighbors_weights):
                    for j,edge in enumerate(neighbor_edges):
                        if abs(edge-maximum_edge)<=eps:
                            if j == 0:
                                filtered_edges[i,0] = edge
                            elif  j == 1:
                                filtered_edges[i,1] = edge
                            elif  j == 2:
                                filtered_edges[i,2] = edge
                            elif  j == 3:
                                filtered_edges[i,3] = edge
            edge_exist_0 = True
            edge_exist_1 = True
            for i,neighbor_edges in enumerate(filtered_edges):
                for j,edge in enumerate(neighbor_edges):
                    if j==0 or j==1:
                        if edge==0:
                            edge_exist_0 = False
                    elif j==2 or j==3:
                        if edge==0:
                            edge_exist_1 = False
            if edge_exist_0 and not edge_exist_1:
                new_label_matrix[i,j] = 0
                eps/=2
            elif not edge_exist_0 and edge_exist_1:
                new_label_matrix[i,j] = 1
                eps/=2
            elif edge_exist_0 or edge_exist_1:
                filtered_edges_0_max = filtered_edges[:,:2].max()
                filtered_edges_1_max = filtered_edges[:,2:].max()
                if filtered_edges_0_max>filtered_edges_1_max:
                    new_label_matrix[i,j] = 0
                    eps/=2
                elif filtered_edges_0_max<filtered_edges_1_max:
                    new_label_matrix[i,j] = 1
                    eps/=2
                elif filtered_edges_0_max==filtered_edges_1_max:
                    new_label_matrix[i,j] = 1
                    eps/=2
            elif not edge_exist_0 or not edge_exist_1:
                new_label_matrix[i,j] = 1
        
    return new_label_matrix, eps

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

def diffusion_sampler(image : np.ndarray,iterations: int = 5) -> Tuple[np.ndarray,List[np.ndarray]]:
    area1 = image[BBOX1[0]:BBOX1[2],BBOX1[1]:BBOX1[3]]
    area1 = area1.reshape(area1.shape[0]*area1.shape[1],3)
    area2 = image[BBOX2[0]:BBOX2[2],BBOX2[1]:BBOX2[3]]
    area2 = area2.reshape(area2.shape[0]*area2.shape[1],3)
    mean,cov_inv,cov_det = get_image_params(area1,area2)
    image = cv2.resize(image,(256,256))
    denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
    results = []
    eps = 0.7
    for iteration in range(iterations):
        label_matrix,eps = compute_obervation(image, mean, denominator, cov_inv, eps)
        print('eps = ',eps)
        print(iteration)
        stacked_img = np.stack((label_matrix,)*3, axis=-1)
        stacked_img[:,:,0]*=255
        cv2.imwrite(f"lab2/results/result_{iteration}.jpg",stacked_img)
        area1 = image[label_matrix==0]
        area2 = image[label_matrix==1]
        mean,cov_inv,cov_det = get_image_params(area1,area2)
        denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
        results.append(label_matrix)
    return label_matrix,results


if __name__ == '__main__':
    image_path = 'lab2/image.jpg'
    image = cv2.imread(image_path)/255.0
    start = time.time()
    seg_image, results = diffusion_sampler(image)
    stacked_img = np.stack((seg_image,)*3, axis=-1)
    stacked_img[:,:,0]*=255
    cv2.imwrite("lab2/result.jpg",stacked_img)
    end = time.time()
    print("TIME = ",end-start)
    for i,mask in enumerate(results):
        stacked_img = np.stack((mask,)*3, axis=-1)*255
        cv2.imwrite(f'lab2/results/image_{i}.jpg',stacked_img)