import numpy as np
import json
import random
import collections
from scipy.spatial import distance
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

with open('results/retention_multimodal.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'architecture', 'distractor_name', 'cap_len', 'token_pos', 'chebyshev', 'cosine', sep='\t', file=f)
with open('results/retention_output.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'architecture', 'distractor_name', 'cap_len', 'token_pos', 'chebyshev', 'cosine', sep='\t', file=f)

with open('results/image_dists.txt', 'r', encoding='utf-8') as f:
    img_dists = { 'mscoco': [] }
    for line in f.read().strip().split('\n')[1:]:
        [dataset_name, _, max_sim_image, _, med_sim_image, _, min_sim_image, _] = line.split('\t')
        img_dists[dataset_name].append({ 'max': int(max_sim_image), 'med': int(med_sim_image), 'min': int(min_sim_image) })

with open('results/caplen_freqs.txt', 'r', encoding='utf-8') as f:
    caplen_freqs = { 'mscoco': [] }
    for line in f.read().strip().split('\n')[1:]:
        [dataset_name, architecture, cap_len, freq] = line.split('\t')
        caplen_freqs[(dataset_name, architecture, int(cap_len))] = int(freq)

for dataset_name in [ 'mscoco' ]:# 'flickr8k', 'flickr30k',
    datasources = helper_datasources.DataSources(dataset_name)
    
    for architecture in [ 'init', 'pre', 'par', 'merge' ]:
        multimodal_entries = collections.defaultdict(lambda:{ 'chebyshev': list(), 'cosine': list() })
        output_entries = collections.defaultdict(lambda:{ 'chebyshev': list(), 'cosine': list() })
        print('{}_{}_{}'.format(dataset_name, architecture, run))
        
        dataset = data.Dataset()
        dataset.minimal_load('model_data/{}_{}_{}'.format(dataset_name, architecture, run))
        
        with model_normal.NormalModel(
                dataset                 = dataset,
                init_method             = config.hyperparams[architecture]['init_method'],
                min_init_weight         = config.hyperparams[architecture]['min_init_weight'],
                max_init_weight         = config.hyperparams[architecture]['max_init_weight'],
                embed_size              = config.hyperparams[architecture]['embed_size'],
                rnn_size                = config.hyperparams[architecture]['rnn_size'],
                post_image_size         = config.hyperparams[architecture]['post_image_size'],
                post_image_activation   = config.hyperparams[architecture]['post_image_activation'],
                rnn_type                = config.hyperparams[architecture]['rnn_type'],
                learnable_init_state    = config.hyperparams[architecture]['learnable_init_state'],
                multimodal_method       = architecture,
                optimizer               = config.hyperparams[architecture]['optimizer'],
                learning_rate           = config.hyperparams[architecture]['learning_rate'],
                normalize_image         = config.hyperparams[architecture]['normalize_image'],
                weights_reg_weight      = config.hyperparams[architecture]['weights_reg_weight'],
                image_dropout_prob      = config.hyperparams[architecture]['image_dropout_prob'],
                post_image_dropout_prob = config.hyperparams[architecture]['post_image_dropout_prob'],
                embedding_dropout_prob  = config.hyperparams[architecture]['embedding_dropout_prob'],
                rnn_dropout_prob        = config.hyperparams[architecture]['rnn_dropout_prob'],
                max_epochs              = config.hyperparams[architecture]['max_epochs'] if not config.debug else 2,
                val_minibatch_size      = config.val_minibatch_size,
                train_minibatch_size    = config.hyperparams[architecture]['train_minibatch_size'],
            ) as m:
            m.compile_model()
            m.load_params('model_data/{}_{}_{}'.format(dataset_name, architecture, run))
            
            images = datasources.test.images
            with open('model_data/{}_{}_{}/generated_captions.txt'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f:
                captions = [ line.split(' ') for line in f.read().strip().split('\n') ]
        
            for (cap, img, distractors) in zip(captions, images, img_dists[dataset_name]):
                if caplen_freqs[dataset_name, architecture, len(cap)] < 20:
                    continue
                
                cap_len = len(cap)+1 #include edge
                cap_prefix = [data.EDGE_INDEX] + [ m.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ]
                for (dist_name, dist_index) in distractors.items():
                    distractor_image = images[dist_index]
                    [ multimodal_vectors, outputs ] = m.session.run(
                        [ m.multimodal_vectors, m.predictions ] ,
                        feed_dict={
                                m.dropout:       False,
                                m.temperature:   1.0,
                                m.prefixes:      [cap_prefix, cap_prefix],
                                m.prefixes_lens: [cap_len, cap_len],
                                m.images:        [img, distractor_image],
                            }
                        )
                    for i in range(cap_len):
                        multimodal_entries[(dist_name, cap_len-1, i)]['chebyshev'].append(distance.chebyshev(multimodal_vectors[0,i], multimodal_vectors[1,i]))
                        multimodal_entries[(dist_name, cap_len-1, i)]['cosine'].append(distance.cosine(multimodal_vectors[0,i], multimodal_vectors[1,i]))
                        
                        output_entries[(dist_name, cap_len-1, i)]['chebyshev'].append(distance.chebyshev(outputs[0,i], outputs[1,i]))
                        output_entries[(dist_name, cap_len-1, i)]['cosine'].append(distance.cosine(outputs[0,i], outputs[1,i]))
                        
        with open('results/retention_multimodal.txt', 'a', encoding='utf-8') as f:
            for (distractor_name, cap_len, token_pos) in sorted(multimodal_entries.keys()):
                dists = multimodal_entries[(distractor_name, cap_len, token_pos)]
                print(dataset_name, architecture, distractor_name, cap_len, token_pos, np.mean(dists['chebyshev']), np.mean(dists['cosine']), sep='\t', file=f)
        with open('results/retention_output.txt', 'a', encoding='utf-8') as f:
            for (distractor_name, cap_len, token_pos) in sorted(output_entries.keys()):
                dists = output_entries[(distractor_name, cap_len, token_pos)]
                print(dataset_name, architecture, distractor_name, cap_len, token_pos, np.mean(dists['chebyshev']), np.mean(dists['cosine']), sep='\t', file=f)