# quantifing-visual-information
Code used by the paper "Quantifying the amount of visual information used by neural caption generators".

This paper is about measuring how important the image is in an image caption generator as it is generating words. Does the image become less important as the caption gets longer? Do all the words being generated depend on the image in similar amounts? We find that word positions that are typically occupied by nouns overwhelmingly depend on the image more than others but that this dependency goes down as the generated caption prefix gets longer. We also find that when analysing the merge architecture (architecture where the image is not injected into the RNN), the logits of the softmax has a smooth and constant image dependency on the image whilst other architectures do not. On the other hand, the image dependencies of the softmax distribution in every architecture are very similar, suggesting that merge architectures are more sensitive to the image but that the softmax function interfere with this sensitivity.

Works on Python 3.

## Dependencies

Python dependencies (install all with `pip`):

* `tensorflow` (v1.4)
* `future`
* `numpy`
* `scipy`
* `h5py`
* `nltk`

## Before running

1. Download [Karpathy's](http://cs.stanford.edu/people/karpathy/deepimagesent/) Flickr8k, Flickr30k, and MSCOCO datasets (including image features).
1. Download the contents of the [where-image2](https://github.com/mtanti/where-image2) results folder into `model_data`. Trained models are not available for immediate download but can be generated by running the experiments from scratch.
1. Open `config.py`.
  1. Set `debug` to True or False (True is used to run a quick test).
  1. Set `raw_data_dir` to return the directory to the Karpathy datasets (`dataset_name` is 'flickr8k', 'flickr30k', or 'mscoco').

## File descriptions

File name    |    Description
---|---
`results`    |    Folder storing results of each image usage. Contents are described separately below.
`get_tags.py` (main)    |    Compute statistics about the distribution of part of speech tags in different word positions in image captions. Results stored in `results/tags.txt`.
`get_caplen_freqs.py` (main)    |    Get frequency of each caption length. Results stored in `results/caplen_freqs.txt`.
`get_img_dists.py` (main)    |    Find the most similar image in the test set to every other image in the test set, the least similar image, and the middle similar image. Results stored in `results/image_dists.txt`.
`analyse_gradients.py` (main)    |    Measure visual information usage using sensitivity analysis. Results stored in `results/gradients.txt`.
`analyse_retention.py` (main)    |    Measure visual information usage using omission scores. Results stored in `results/retention_multimodal.txt` and `results/retention_output.txt`.
`analyse_retention_logits.py` (main)    |    Like `analyse_retention.py` but performs the analysis on the logits. Results stored in `results/retention_logits.txt`.
`analyse_retention_output_jsd.py` (main)    |    Like `analyse_retention.py` but uses Jensen-Shannon Divergence on the softmax instead of cosine distance. Results stored in `results/retention_output_jsd.txt`.
`analyse_logits_stats.py` (main)    |    Compute statistics about the logits. Results stored in `results/logits_stats.txt`.
`results.xlsx` (processed data)    |    MS Excel spreadsheet with the results.

Other files are copied from [where-image2](https://github.com/mtanti/where-image2) with some modification to be able to calculate the gradient of the output with respect to the image.