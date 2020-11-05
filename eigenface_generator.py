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
        """
        Generates all the needed datasets in "eigen_faces_npy" folder (all datasets are numpy Arrays.)
        If folder "eigen_faces_npy" does not exist, it will be created next to "images" folder.
        """

        # creating "/eigen_faces_npy" folder if it does not exit
        if not (os.path.exists(self.path_eigen_faces_npy)):
            os.makedirs(self.path_eigen_faces_npy)

        self._generate_eigenface()
        print('_generate_eigenface() ok')
        self._generate_eigenface_gallery_pictures()
        print('_generate_eigenface_gallery_pictures() ok')
        self._generate_eigenface_probes_pictures()
        print('_generate_eigenface_probes_pictures() ok')

    def _generate_eigenface(self):
        """
        Generate de eigenface vectors and the average face of the gallery_picture_dataset
        The .npy files will be stored in the "eigen_faces_npy" folder
        :return:
        """

        # We flatten each (150x150) pictures into a list of 22500 pixels
        # Then we put all the flat pictures one next to the other in a numpy array
        # Each column represent a pixel and each line represents a picture

        linear_pictures = np.array([picture.flatten() for person in self.gallery_pictures for picture in person])

        # As each column is a pixel, we need to compute the average column for all pictures
        mean_face = linear_pictures.mean(axis=0)

        # subtract mean face to every picture
        linear_pictures_centered = linear_pictures - mean_face

        # To make PCA faster, we transpose it -> each line is a pixel and each column is a face picture
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

        # the eigen faces are computing by multiplying the transposed linear_pictures_centered matrix and the components
        # of the PCA and then, we need to make their norm = 1

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
        """
         Get numpy arrays of the eigenfaces and the mean face
         :return: eigen_faces -> array of eigen_faces vectors
         mean_face -> a flat representation of the average face
         """
        if not (os.path.exists(self.path_eigen_faces_npy + "/eigen_faces.npy")):
            raise Exception("eigen_faces.npy does not exist, please call _generate_eigenface() first.")
        if not (os.path.exists(self.path_eigen_faces_npy + "/mean_face.npy")):
            raise Exception("mean_face.npy does not exist, please call _generate_eigenface() first.")

        eigen_faces = np.load(self.path_eigen_faces_npy + "/eigen_faces.npy", allow_pickle=True)
        mean_face = np.load(self.path_eigen_faces_npy + "/mean_face.npy", allow_pickle=True)
        return eigen_faces, mean_face

    def _generate_eigenface_gallery_pictures(self):
        """
         Generate de eigenface coefficients of the gallery_pictures_dataset
         The .npy file will be stored in the "eigen_faces_npy" folder

         coefficients_list = (flat_picture - mean_face) * transposed_eigen_faces_vectors
         There are as many coefficients as the number of eigen_faces_vectors
         :return:
         """

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
        """
         Get numpy arrays of the eigenfaces coefficients of the gallery_picture dataset
         :return: gallery_eigenface_pictures -> array of arrays. Each subarray corresponds to a list of 1 person's
         pictures. Here the pictures are described as a list of the coefficient of the eigenfaces
         """
        if not (os.path.exists(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy")):
            raise Exception("gallery_eigenface_pictures.npy does not exist,"
                            " please call _generate_gallery_eigenface_pictures() first.")

        gallery_eigenface_pictures = np.load(self.path_eigen_faces_npy + "/gallery_eigenface_pictures.npy",
                                             allow_pickle=True)
        return gallery_eigenface_pictures

    def _generate_eigenface_probes_pictures(self):
        """
         Generate de eigenface coefficients of the probes_eigenface_pictures
         The .npy file will be stored in the "eigen_faces_npy" folder

         coefficients_list = (flat_picture - mean_face) * transposed_eigen_faces_vectors
         There are as many coefficients as the number of eigen_faces_vectors
         :return:
         """

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
        """
         Get numpy arrays of the eigenfaces coefficients of the probes_eigenface_pictures
         :return: probes_eigenface_pictures -> array : each array's element corresponds to one list of eigenfaces
         coefficients
         """
        if not (os.path.exists(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy")):
            raise Exception("probes_eigenface_pictures.npy does not exist,"
                            " please call _generate_eigenface_probes_pictures() first.")

        probes_eigenface_pictures = np.load(self.path_eigen_faces_npy + "/probes_eigenface_pictures.npy",
                                            allow_pickle=True)
        return probes_eigenface_pictures

    def get_coefs_from_picture(self, picture):
        """
        Computes a list of eigen coefficients for a given 2D picture.
        :param picture: 2D numpy array
        :return: 1D numpy array as long as the number of eigenfaces vectors list
        """
        eigen_faces, mean_face = self.get_eigenfaces()
        picture_flat_center = picture.flatten() - mean_face

        return np.dot(picture_flat_center, np.transpose(eigen_faces))

    def get_picture_from_coefs(self, coefs):
        """
        Generates a flat picture (1D) from a given list of eigen coefficients.
        :param coefs: list of eigen coefficients
        :return: 1D numpy array representing a 1D picture
        """
        eigen_faces, mean_face = self.get_eigenfaces()
        return mean_face + np.dot(coefs, eigen_faces)
