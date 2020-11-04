from gallery_probes_generator import GalleryProbesGenerator

import numpy as np


data_generator = GalleryProbesGenerator(path_to_dataset='data/dataset1/', probes_length=200)

data_generator.generate_npy_files()

probe_registered_names, probe_registered_pictures = data_generator.get_registered_probes()
probe_unregistered_names, probe_unregistered_pictures = data_generator.get_unregistered_probes()
gallery_names, gallery_pictures = data_generator.get_gallery()
probe_merged_names, probe_merged_pictures = data_generator.get_merged_probes()

print("hello")

# val_propres = np.load('data/dataset1/eigenface_npy/val_propres.npy', allow_pickle=True)
# vect_propres = np.load('data/dataset1/eigenface_npy/vect_propres.npy', allow_pickle=True)
# w = np.load('data/dataset1/eigenface_npy/w.npy', allow_pickle=True)