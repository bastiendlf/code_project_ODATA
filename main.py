from gallery_probes_generator import GalleryProbesGenerator


data_generator = GalleryProbesGenerator(path_to_dataset='data/dataset1/', probes_length=200)

data_generator.generate_npy_files()

probe_registered_names, probe_registered_pictures = data_generator.get_registered_probes()
probe_unregistered_names, probe_unregistered_pictures = data_generator.get_unregistered_probes()
gallery_names, gallery_pictures = data_generator.get_gallery()