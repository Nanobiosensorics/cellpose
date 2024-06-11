import os, re

def get_image_ids(path):
    pattern = r'^(\d+[_-]\d+(?:[_-]\d+)?)'
    if os.path.isdir(path):
        ids = []
        for filename in os.listdir(path):
            match = re.match(pattern, filename)
            if match is not None:
                ids.append(match.group(1))
        ids = sorted(list(set(ids)))
    else:
        split = path.split(os.sep)
        if len(split) <= 1:
            split = path.split('/')
        parts, name = split[:-1], split[-1]
        name = re.match(pattern, name)
        if name is None:
            raise Exception('Experiment id not found in filename!')
        ids = [name.group(1)]
        path = os.path.join(*parts)
        if os.sep == '/':
            path = '/' + path
    return path, ids

def get_tif(path, idx):
    files = [name for name in os.listdir(path) if name.endswith('5.tif') and idx in name]
    return None if len(files) == 0 else files[0]

def get_smlm_file(path, idx):
    files = [name for name in os.listdir(path) if name.endswith('.txt') and idx in name]
    return None if len(files) == 0 else files[0]

def get_smlm_aligned_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'ClusterData' in name]
    return None if len(files) == 0 else files[0]

def get_srrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'segmResultsPRED' in name]
    return None if len(files) == 0 else files[0]

def get_esrrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and name.endswith('_esrrf.tif')]
    return None if len(files) == 0 else files[0]

def get_seg_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'seg.npy' in name]
    return None if len(files) == 0 else files[0]

def get_raw_srrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and name.endswith('.ome.tif')]
    return None if len(files) == 0 else files[0]

def get_sample(path, idx):
    if os.path.isfile(path):
        path = os.path.join(*path.split('/')[:-1])
    return { 
            'img' : os.path.join(path, f) if (f := get_tif(path, idx)) is not None else None, 
            'smlm': os.path.join(path, f) if (f := get_smlm_file(path, idx)) is not None else None, 
            'smlm_aligned': os.path.join(path, f) if (f := get_smlm_aligned_file(path, idx)) is not None else None, 
            'srrf': os.path.join(path, f) if (f := get_srrf_file(path, idx)) is not None else None,
            'esrrf':os.path.join(path, f) if (f := get_esrrf_file(path, idx)) is not None else None,
            'raw-srrf': os.path.join(path, f) if (f := get_raw_srrf_file(path, idx)) is not None else None,
            'seg': os.path.join(path, f) if (f := get_seg_file(path, idx)) is not None else None
            }