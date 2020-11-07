from eigenface_generator import EigenfaceGenerator
from gallery_probes_generator import GalleryProbesGenerator

path_to_datasets = ['data/dataset1/', 'data/dataset2/']

for path in path_to_datasets:
    print(f'-----Generating files in {path}-----')
    print(f'Generating gallery and probes with GalleryProbesGenerator...')
    data_generator = GalleryProbesGenerator(path_to_dataset=path, probes_length=200)
    data_generator.generate_npy_files()
    print(f'Generating eigenfaces with EigenfaceGenerator...')
    eigenfacegenerator = EigenfaceGenerator(data_generator=data_generator, components=500)
    eigenfacegenerator.generate_npy_files()
    print(f'NPY files generation in {path} completed.')
