import numpy as np
import cv2
import matplotlib
from typing import Tuple,Any,List
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


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
    return (numerator/denominator)


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


def remove_bones(neighbors_weights,eps):
    filtered_edges = np.zeros((len(neighbors_weights),4))
    for i,neighbor_edges in enumerate(neighbors_weights):
        maximum_edge = max(neighbor_edges)
        for j,edge in enumerate(neighbor_edges):
            if edge-maximum_edge<=eps:
                filtered_edges[i,j] = edge
            else:
                filtered_edges[i,j] = None
        for n, neighbor_edges in enumerate(neighbors_weights[:,:2]):
            selected = list(set(range(len(neighbors_weights[:,:2]))) - set([n]))
            nonNaN_bones_amount = len(np.argwhere(np.isnan(selected) == False))
            if nonNaN_bones_amount==0:
                filtered_edges[n,0] = None
                filtered_edges[n,1] = None
        for n, neighbor_edges in enumerate(neighbors_weights[:,2:]):
            selected = list(set(range(len(neighbors_weights[:,2:]))) - set([n]))
            nonNaN_bones_amount = len(np.argwhere(np.isnan(selected) == False))
            if nonNaN_bones_amount==0:
                filtered_edges[n,2] = None
                filtered_edges[n,3] = None
    return filtered_edges



def compute_obervation(
    image : np.ndarray,
    mean : Tuple[np.ndarray], 
    denominator : Tuple[float],
    cov_inv :  Tuple[np.ndarray],
    eps_matrix : np.ndarray) -> np.ndarray:
    new_label_matrix = np.zeros((image.shape[0],image.shape[1]))
    for i in tqdm(range(image.shape[0]-1)):
        for j in range(image.shape[1]-1):
            neighbors = get_neighbors(image,i,j)
            neighbors_weights = []
            for _ in neighbors:
                p0_0 = np.log(gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0])) + np.log(0.7)
                p0_0 = p0_0 if p0_0!=-np.inf else -1000
                p0_1 = np.log(gauss_prob(image[i,j], mean[0], denominator[0], cov_inv[0])) + np.log(0.3)
                p0_1 = p0_1 if p0_1!=-np.inf else -1000
                p1_0 = np.log(gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1])) + np.log(0.3)
                p1_0 = p1_0 if p1_0!=-np.inf else -1000
                p1_1 = np.log(gauss_prob(image[i,j], mean[1], denominator[1], cov_inv[1])) + np.log(0.7)
                p1_1 = p1_1 if p1_1!=-np.inf else -1000
                added = [p0_0,p0_1,p1_0,p1_1]
                min_added = abs(min(added))
                weight_in_neighbor = []
                for el in added:
                    weight_in_neighbor.append(round(el+min_added,3))
                del added
                neighbors_weights.append(weight_in_neighbor)
            neighbors_weights = np.array(neighbors_weights)
            for _ in range(100):
                overweights = []
                for _,neighbor_edges in enumerate(neighbors_weights):
                    max_edges_0 = neighbor_edges[:2].max()
                    max_edges_1 = neighbor_edges[2:].max()
                    sum_max_edges_0 = sum(neighbors_weights[:,:2].max(axis = 1))
                    sum_max_edges_1 = sum(neighbors_weights[:,2:].max(axis = 1))
                    overweighted_neighbor = []
                    for l,edge in enumerate(neighbor_edges):
                        if l == 0 or l==1:
                            new_edge = edge - max_edges_0 + (sum_max_edges_0/neighbors_weights.shape[0])
                        elif l == 2 or l==3:
                            new_edge = edge - max_edges_1 + (sum_max_edges_1/neighbors_weights.shape[0])
                        overweighted_neighbor.append(new_edge)
                    overweights.append(overweighted_neighbor)
                neighbors_weights = np.array(overweights)
            filtered_edges = remove_bones(neighbors_weights,eps_matrix[i][j])
            while np.isnan(filtered_edges.max()):
                eps_matrix[i][j] = eps_matrix[i][j]*1.25
                filtered_edges = remove_bones(neighbors_weights,eps_matrix[i][j])
            else:
                eps_matrix[i][j] = eps_matrix[i][j]/2
                max_bone = np.argwhere(filtered_edges == filtered_edges.max())[0]
                if max_bone[1] == 0 or max_bone[1] == 1:
                    new_label_matrix[i][j] = 0
                elif max_bone[1] == 2 or max_bone[1] == 3:
                    new_label_matrix[i][j] = 1
        
    return new_label_matrix, eps_matrix

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
    eps_matrix = np.ones(image.shape[:2])*100
    for iteration in range(iterations):
        label_matrix,eps_matrix = compute_obervation(image, mean, denominator, cov_inv,eps_matrix)
        print(iteration)
        stacked_img = np.stack((label_matrix,)*3, axis=-1)
        stacked_img[:,:,0]*=255
        cv2.imwrite(f"results/result_{iteration}.jpg",stacked_img)
        area1 = image[label_matrix==0]
        area2 = image[label_matrix==1]
        mean,cov_inv,cov_det = get_image_params(area1,area2)
        denominator = [np.sqrt(((2 * np.pi)**3) * cov_det[0]),np.sqrt(((2 * np.pi)**3) * cov_det[1])]
        results.append(label_matrix)
    return label_matrix,results


if __name__ == '__main__':
    image_path = 'image.jpg'
    image = cv2.imread(image_path)/255.0
    start = time.time()
    seg_image, results = diffusion_sampler(image)
    stacked_img = np.stack((seg_image,)*3, axis=-1)
    stacked_img[:,:,0]*=255
    cv2.imwrite("result.jpg",stacked_img)
    end = time.time()
    print("TIME = ",end-start)
    for i,mask in enumerate(results):
        stacked_img = np.stack((mask,)*3, axis=-1)*255
        cv2.imwrite(f'results/image_{i}.jpg',stacked_img)