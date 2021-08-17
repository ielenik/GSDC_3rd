from loader import NaN
from tqdm import tqdm
import datetime
import pickle
import os
import numpy as np
import pathlib
import coords_tools as ct
import matplotlib.pyplot as plt

INPUT = 'data/'
p = pathlib.Path(INPUT)

def load_ground_truth_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    lines = lines[1:]
    times = []
    obs = []
    for l in lines:
        sp = l.split(',')
        times.append(int(sp[2]))
        obs.append(np.array(sp[3:]).astype(np.float))

    return np.array(times), np.array(obs)

def load_all_truth():
    train_files = list(p.glob('train/*/*/ground_truth.csv'))

    times = []
    obs = []
    for f in tqdm(train_files):
        t,o = load_ground_truth_file(f)
        if len(times) == 0:
            times = t
            obs = o
        else:
            times = np.concatenate((times,t))
            obs = np.concatenate((obs,o))
    
    obs[:,0],obs[:,1],obs[:,2] = ct.WGS84_to_ECEF(obs[:,0],obs[:,1],obs[:,2])
    
    indexes = np.argsort(obs[:,0]) 
    indexes = indexes[len(indexes)*45//100:len(indexes)*55//100]
    obs_sort = obs[indexes]

    indexes = np.argsort(obs_sort[:,1]) 
    indexes = indexes[len(indexes)*45//100:len(indexes)*55//100]
    obs_sort = obs_sort[indexes]

    indexes = np.argsort(obs_sort[:,2]) 
    center = obs_sort[indexes[len(indexes)//2]]
    
    zvect = center[:3]/np.linalg.norm(center[:3])
    xvect = np.array([1,0,0])
    xvect = xvect - np.sum(xvect*zvect)*zvect
    xvect = xvect/np.linalg.norm(xvect, axis = -1, keepdims=True)
    yvect = np.cross(zvect, xvect)

    mat = np.transpose(np.stack((xvect,yvect,zvect)))

    obs[:,:3] -= center[:3]

    norms = np.linalg.norm(obs_sort[:,:3], axis = -1)
    print(np.min(norms))
    print(np.max(norms))    

    obs[:,:3] = np.matmul(obs[:,:3],mat)

    print(np.min(obs,axis=0))
    print(np.max(obs,axis=0))    

    img_min = np.full((1024,1024),100000.)
    img_max = np.full((1024,1024),-100000.)
    xoords = np.array(1023*(obs[:,0]-np.min(obs[:,0]))/(np.max(obs[:,0])-np.min(obs[:,0]))).astype(int)
    yoords = np.array(1023*(obs[:,1]-np.min(obs[:,1]))/(np.max(obs[:,1])-np.min(obs[:,1]))).astype(int)
    zoords = obs[:,2]

    for i in range(len(xoords)):
        img_min[xoords[i],yoords[i]] = min(img_min[xoords[i],yoords[i]],zoords[i])
        img_max[xoords[i],yoords[i]] = max(img_min[xoords[i],yoords[i]],zoords[i])
    img_max -= img_min

    img_max[img_max == -200000] = 0
    print(np.min(img_max))
    print(np.max(img_max))
    img_max = (img_max*25).astype(np.uint8)
    plt.imshow(img_max)
    plt.show()

    '''
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.scatter(obs[:,0], obs[:,1], obs[:,2])
    plt.show()
    '''