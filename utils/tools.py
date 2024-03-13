# utils/tools.py

import numpy as np

def background_del(seg_with_back):
    # Determine if the values of each channel in RGB are the same, and if they are, convert them to (0, 0, 0).
    same_channels = np.all(seg_with_back[..., :-1] == seg_with_back[..., 1:], axis=-1)
    seg_with_back[same_channels] = (0, 0, 0)
    
    return seg_with_back, ~same_channels

def assign_cluster_number(seg, seg_bool):
    
    cluster = np.zeros_like(seg_bool, dtype=np.uint16)
    color = [[0,0,0]]
    
    nonzero_indices = np.nonzero(seg_bool)
    for y, x in zip(*nonzero_indices):
        c = seg[y, x]  # The color of the target pixel.
    
        result_c = np.all(c == np.array(color), axis=-1)
    
        if np.any(result_c):
            cluster[y, x] = np.argmax(result_c)
        else:
            color.append(list(c))
            cluster[y, x] = len(color)-1
    
    return cluster, color