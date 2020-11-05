import numpy as np
import os
from sklearn.decomposition import PCA


class EigenfaceGenerator:

    def __init__(self, data_generator, components=500):
        self.path_to_dataset = data_generator.path_to_dataset
        self.path_dataset_npy = self.path_to_dataset + '/npy'
        self.path_eigen_faces_npy = self.path_to_dataset + '/eigen_faces_npy'

        self.components = components
        self.variance_kept = 0

        self.data_generator = data_generator
        self.gallery_pictures = data_generator.get_gallery()[1]
        self.probes_pictures = data_generator.get_merged_probes()[1]

    def generate_npy_files(self):

        # creating "npy" folder if it does not exit
        if not (os.path.exists(self.path_eigen_faces_npy)):
            os.makedirs(self.path_eigen_faces_npy)

        self._generate_eigenface()
        print('_generate_eigenface() ok')
        self._generate_eigenface_gallery_pictures()
        print('_generate_eigenface_gallery_pictures() ok')
        self._generate_eigenface_probes_pictures()
        print('_generate_eigenface_probes_pictures() ok')

    def _generate_eigenface(self):
        linear_pictures = np.array([picture.flatten() for person in self.gallery_pictures for picture in person])
        mean_face = linear_pictures.mean(axis=0)

        # subtract mean face to every picture
        linear_pictures_centered = linear_pictures - mean_face

        training_data = np.transpose(linear_pictures_centered)
        pca = PCA(n_components=self.components).fit(training_data)
        self.variance_kept = sum(pca.explained_variance_ratio_)
        print(f"Variance kept :{self.variance_kept}")

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        print('Computing eigenfaces...')
        transpose_linear_pictures_centered = np.transpose(linear_pictures_centered)
        eigen_faces = list()
        for vector in pca.components_:
            eigen_faces.append(normalize(np.dot(transpose_linear_pictures_centered, vector)))

        eigen_faces = np.array(eigen_faces)

        #  remove the previous generated datasets
        if os.path.exists(self.path_eigen_faces_npy + "/eigen_faces.npy"):
            print('Removing previous eigen_faces.npy ...')
            os.remove(self.path_eigen_faces_npy + "/eigen_faces.npy")

        if os.path.exists(self.path_eigen_faces_npy + "/mean_face.npy"):
            print('Removing previous mean_face.npy ...')
            os.remove(self.path_eigen_faces_npy + "/mean_face.npy")

        # saving eigen_face in folder
        print(f'Saving eigen_faces and mean_face in {self.path_eigen_faces_npy}')
        np.save(self.path_eigen_faces_npy + "/eigen_faces.npy", eigen_faces)
        np.save(self.path_eigen_faces_npy + "/mean_face.npy", mean_face)

    def get_eigenfaces(self):
        if not (os.path.exists(self.path_eigen_faces_npy + "/eigen_faces.npy")):
            raise Exception("eigen_faces.npy does not exist, please call _generate_eigenface() first.")
        if not (os.path.exists(self.path_eigen_faces_npy + "/mean_face.npy")):
            raise Exception("mean_face.npy does not exist, please call _generate_eigenface() first.")

        eigen_faces = np.load(self.path_eigen_faces_npy + "/eigen_faces.npy", allow_pickle=True)
        mean_face = np.load(self.path_eigen_faces_npy + "/mean_face.npy", allow_pickle=True)
        return eigen_faces, mean_face

    def _generate_eigenface_gallery_pictures(self):

        eigen_faces, mean_face = self.get_eigenfaces()

        transpose_eigen_faces = np.transpose(eigen_faces)

        gallery_eigenface_pictures = list()
        print("Computing new gallery with eignface...")
        for person in self.gallery_pictures:
            list_of_pictures_coefs = list()

            for picture in person:
                picture = picture.flatten() - mean_face
                list_of_pictures_coefs.append(np.dot(picture, transpose_eigen_faces))
            gallery_eigenface_pictures.append(np.array(list_of_pictures_coefs))

        gallery_eigenface_pictures = np.array(gallery_eigenface_pictures, dtype=object)

        if os.path.exists(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy"):
            print('Removing previous gallery_eigenface_pictures.npy ...')
            os.remove(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy")

        # saving new gallery_picture in folder in folder
        print(f'Saving gallery_eigenface_pictures in {self.path_eigen_faces_npy}')
        np.save(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy", gallery_eigenface_pictures)

    def get_gallery_picture_eigenface(self):
        if not (os.path.exists(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy")):
            raise Exception("gallery_eigenface_pictures.npy does not exist,"
                            " please call _generate_gallery_eigenface_pictures() first.")

        gallery_eigenface_pictures = np.load(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy",
                                             allow_pickle=True)
        return gallery_eigenface_pictures

    def _generate_eigenface_probes_pictures(self):

        eigen_faces, mean_face = self.get_eigenfaces()
        transpose_eigen_faces = np.transpose(eigen_faces)

        probes_eigenface_pictures = list()
        for picture in self.probes_pictures:
            picture = picture.flatten() - mean_face
            probes_eigenface_pictures.append(np.dot(picture, transpose_eigen_faces))

        probes_eigenface_pictures = np.array(probes_eigenface_pictures)

        if os.path.exists(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy"):
            print('Removing previous probes_eigenface_pictures.npy ...')
            os.remove(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy")

        # saving probes_eigenface_pictures in folder in folder
        print(f'Saving probes_eigenface_pictures in {self.path_eigen_faces_npy}')
        np.save(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy", probes_eigenface_pictures)

    def get_probes_pictures_eigenface(self):
        if not (os.path.exists(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy")):
            raise Exception("probes_eigenface_pictures.npy does not exist,"
                            " please call _generate_eigenface_probes_pictures() first.")

        probes_eigenface_pictures = np.load(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy",
                                            allow_pickle=True)
        return probes_eigenface_pictures

    def get_coefs_from_picture(self, picture):
        """
        pierre = linear_pictures_centered[105]
        alpha = np.dot(pierre, np.transpose(eigen_faces))
        """
        eigen_faces, mean_face = self.get_eigenfaces()
        picture_flat_center = picture.flatten() - mean_face

        return np.dot(picture_flat_center, np.transpose(eigen_faces))

    def get_picture_from_coefs(self, coefs):
        """
        pierrot_le_s = meanPoint + np.dot(new_gallery_pictures[270][18], eigen_faces)
        """
        eigen_faces, mean_face = self.get_eigenfaces()
        return mean_face + np.dot(coefs, eigen_faces)
