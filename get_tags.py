import numpy as np
import json
import collections
import nltk
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

with open('results/tags.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'architecture', 'cap_len', 'token_pos', 'tag', 'freq', 'prop', sep='\t', file=f)
    for dataset_name in ['mscoco']:
        for architecture in [ 'init', 'pre', 'par', 'merge' ]:
            with open('model_data/{}_{}_{}/generated_captions.txt'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f2:
                captions = [ line.split(' ') for line in f2.read().strip().split('\n') ]
            
            tag_freqs = collections.defaultdict(lambda:collections.defaultdict(lambda:0))
            for cap in captions:
                token_tags = nltk.pos_tag(cap, tagset='universal')
                for i in range(len(token_tags)):
                    tag_freqs[(len(cap), i)][token_tags[i][1]] += 1

            for ((cap_len, token_pos), freqs) in sorted(tag_freqs.items()):
                total_freq = sum(freqs.values())
                for (tag, freq) in freqs.items():
                    print(dataset_name, architecture, cap_len, token_pos, tag, freq, freq/total_freq, sep='\t', file=f)