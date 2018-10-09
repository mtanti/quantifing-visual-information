import numpy as np
import json
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import lib
import model_base
import model_normal
import data
import helper_datasources
import config

########################################################################################
run = 1

with open('results/caplen_freqs.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'architecture', 'cap_len', 'freq', sep='\t', file=f)

    for dataset_name in ['mscoco']:
        for architecture in [ 'init', 'pre', 'par', 'merge' ]:
            with open('model_data/{}_{}_{}/generated_captions.txt'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f:
                captions = [ line.split(' ') for line in f.read().strip().split('\n') ]
            
            freqs = collections.defaultdict(lambda:0)
            for cap in captions:
                freqs[len(cap)] += 1

            for (cap_len, freq) in sorted(freqs.items()):
                print(dataset_name, architecture, cap_len, freq, sep='\t', file=f)