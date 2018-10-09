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

with open('results/caplen_freqs.txt', 'r', encoding='utf-8') as f:
    caplen_freqs = { 'mscoco': [] }
    for line in f.read().strip().split('\n')[1:]:
        [dataset_name, architecture, cap_len, freq] = line.split('\t')
        caplen_freqs[(dataset_name, architecture, int(cap_len))] = int(freq)

with open('results/gradients.txt', 'w', encoding='utf-8') as f:
    print('dataset', 'architecture', 'cap_len', 'token_pos', 'mean_img_grad', 'mean_prevtoken_grad', sep='\t', file=f)

for dataset_name in [ 'mscoco' ]:# 'flickr8k', 'flickr30k',
    datasources = helper_datasources.DataSources(dataset_name)
    
    for architecture in [ 'init', 'pre', 'par', 'merge' ]:
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
            ) as model:
            model.compile_model()
            model.load_params('model_data/{}_{}_{}'.format(dataset_name, architecture, run))
            
            with open('model_data/{}_{}_{}/generated_captions.txt'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f:
                captions = [ line.split(' ') for line in f.read().strip().split('\n') ]
            
            unpadded_caps = []
            cap_lens = []
            images = []
            for (cap, img) in zip(captions, datasources.test.images):
                if caplen_freqs[dataset_name, architecture, len(cap)] < 20:
                    continue
                    
                unpadded_caps.append([ model.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ])
                cap_lens.append(len(cap)+1) #include edge
                images.append(img)
                
            max_len = max(cap_lens)
            padded_prefixes = np.zeros([len(unpadded_caps), max_len], np.int32)
            padded_targets = np.zeros([len(unpadded_caps), max_len], np.int32)
            for (k, (indexes, cap_len)) in enumerate(zip(unpadded_caps, cap_lens)):
                padded_prefixes[k,:cap_len] = [data.EDGE_INDEX]+indexes
                padded_targets[k,:cap_len] = indexes+[data.EDGE_INDEX]
            
            img_grads_by_token = collections.defaultdict(list)
            prevtoken_grads_by_token = collections.defaultdict(list)
            batch_indexes = list()
            batch_prefixes = list()
            batch_prefix_lens = list()
            batch_images = list()
            batch_targets = list()
            for curr_prefix_len in range(1, max_len+1):
                for (index, (prefix, prefix_len, image, target)) in enumerate(zip(padded_prefixes, cap_lens, images, padded_targets)):
                    if prefix_len < curr_prefix_len:
                        continue
                    
                    batch_indexes.append(index)
                    batch_prefixes.append(prefix[:curr_prefix_len])
                    batch_prefix_lens.append(prefix_len)
                    batch_images.append(image)
                    batch_targets.append(target[:curr_prefix_len])
                    if len(batch_indexes) == model.val_minibatch_size:
                        [img_target_grads, prevtoken_target_grads] = model.session.run(
                            [model.grad_wrt_img, model.grad_wrt_prevtoken],
                            feed_dict={
                                    model.dropout:       False,
                                    model.temperature:   1.0,
                                    model.prefixes:      batch_prefixes,
                                    model.prefixes_lens: batch_prefix_lens,
                                    model.images:        batch_images,
                                    model.targets:       batch_targets
                                }
                            )
                        for (prefix_len, img_grads, prevtoken_grads) in zip(batch_prefix_lens, img_target_grads, prevtoken_target_grads):
                            img_grads_by_token[(prefix_len, curr_prefix_len)].append(np.abs(img_grads))
                            prevtoken_grads_by_token[(prefix_len, curr_prefix_len)].append(np.abs(prevtoken_grads))
                            
                        batch_indexes.clear()
                        batch_prefixes.clear()
                        batch_prefix_lens.clear()
                        batch_images.clear()
                        batch_targets.clear()
                
                if len(batch_indexes) > 0:
                    [img_target_grads, prevtoken_target_grads] = model.session.run(
                        [model.grad_wrt_img, model.grad_wrt_prevtoken],
                        feed_dict={
                                model.dropout:       False,
                                model.temperature:   1.0,
                                model.prefixes:      batch_prefixes,
                                model.prefixes_lens: batch_prefix_lens,
                                model.images:        batch_images,
                                model.targets:       batch_targets
                            }
                        )
                    for (prefix_len, img_grads, prevtoken_grads) in zip(batch_prefix_lens, img_target_grads, prevtoken_target_grads):
                        img_grads_by_token[(prefix_len, curr_prefix_len)].append(np.abs(img_grads))
                        prevtoken_grads_by_token[(prefix_len, curr_prefix_len)].append(np.abs(prevtoken_grads))
                        
                    batch_indexes.clear()
                    batch_prefixes.clear()
                    batch_prefix_lens.clear()
                    batch_images.clear()
                    batch_targets.clear()
        
        with open('results/gradients.txt', 'a', encoding='utf-8') as f:
            for (cap_len, token_pos) in sorted(img_grads_by_token.keys()):
                print(
                    dataset_name,
                    architecture,
                    cap_len-1,
                    token_pos-1,
                    np.mean([np.mean(img_grads) for img_grads in img_grads_by_token[(cap_len, token_pos)]]),
                    np.mean([np.mean(prevtoken_grads) for prevtoken_grads in prevtoken_grads_by_token[(cap_len, token_pos)]]),
                    sep='\t', file=f
                )