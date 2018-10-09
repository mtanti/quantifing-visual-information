import numpy as np
import json
import collections
import os
from scipy.spatial import distance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import lib
import model_base
import model_normal
import data
import helper_datasources
import config

########################################################################################
with open('results/image_dists.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'image', 'max_sim_image', 'max_sim_image_dist', 'med_sim_image', 'med_sim_image_dist', 'min_sim_image', 'min_sim_image_dist', sep='\t', file=f)
    for dataset_name in [ 'mscoco' ]:# 'flickr8k', 'flickr30k',
        datasources = helper_datasources.DataSources(dataset_name)
        
        images = datasources.test.images
        
        for i in range(len(images)):
            print(i+1, len(images))
            dists = [ distance.cosine(images[i], images[j]) for j in range(len(images)) if j != i ]
            sorted_indexes = np.argsort(dists)
            (max_index, med_index, min_index) = (sorted_indexes[0], sorted_indexes[len(sorted_indexes)//2], sorted_indexes[-1])
            print(dataset_name, i, max_index, dists[max_index], med_index, dists[med_index], min_index, dists[min_index], sep='\t', file=f)