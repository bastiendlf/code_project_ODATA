from gallery_probes_generator import GalleryProbesGenerator
from eigenface_generator import EigenfaceGenerator
import matplotlib.pyplot as plt
import numpy as np

data_generator = GalleryProbesGenerator(path_to_dataset='data/dataset1/', probes_length=200)

# data_generator.generate_npy_files()


gallery_names, gallery_pictures = data_generator.get_gallery()
probe_merged_names, probe_merged_pictures = data_generator.get_merged_probes()


eigenfacegenerator = EigenfaceGenerator(data_generator=data_generator, components=500)

# eigenfacegenerator.generate_npy_files()

# gallery_eigenface_pictures = eigenfacegenerator.get_gallery_picture_eigenface()
# probes_eigenface_pictures = eigenfacegenerator.get_probes_pictures_eigenface()
# eigen_faces, mean_face = eigenfacegenerator.get_eigenfaces()
#

person = 170
picture = 15

pic = gallery_pictures[person][picture]
personn_pic = plt.imshow(pic, cmap="gray")
plt.show()

coefs_pic = eigenfacegenerator.get_coefs_from_picture(pic)
created_picture = np.reshape(eigenfacegenerator.get_picture_from_coefs(coefs_pic), (-1, 150))

created_pic = plt.imshow(created_picture, cmap="gray")
plt.show()