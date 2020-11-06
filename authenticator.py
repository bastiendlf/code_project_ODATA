from gallery_probes_generator import GalleryProbesGenerator
from eigenface_generator import EigenfaceGenerator
import numpy as np


class Authenticator:
    def __init__(self, data_generator: GalleryProbesGenerator, eigen_face_generator: EigenfaceGenerator):
        self.gallery_names, self.gallery_pictures = data_generator.get_gallery()
        self.probe_names, self.probe_pictures = data_generator.get_merged_probes()
        self.ground_truth = data_generator.get_ground_truth()

        self.radius = 1970000

    def authenticate(self, probe_image, gallery_images):
        """
        Accepts or denies access to a probe image.
        :param probe_image: unknown image in gallery (could be eigenface coefficients)
        :param gallery_images: gallery of registered people pictures (could be eigenface coefficients)
        :return: Tuple -> (Boolean, list(IDs_of_people_in_gallery_similar_enough)
        If the access is Accepted -> (True, [id_person#1, id_person#2, ...]
        If the access is Denied -> (False, None)
        """
        search_among_gallery = radius_search_bruteforce(data=gallery_images, q=probe_image, radius=self.radius)
        result = (False, None)
        if len(search_among_gallery) > 0:  # At least one neighbor found -> authentication accepted
            id_neighbors_found = list()
            for element in search_among_gallery:
                if element[0] not in id_neighbors_found:
                    id_neighbors_found.append(element[0])
            result = (True, np.array(id_neighbors_found))
        return result

    def authenticate_all_probes(self, gallery_images, probes_images):
        """
        Accepts or denies access to a set of probes images.
        :param probes_images: set of unknown images in gallery (could be eigenface coefficients)
        :param gallery_images: gallery of registered people pictures (could be eigenface coefficients)
        :return: List of tuples, index corresponding to the probes order
        Tuple -> (Boolean, list(IDS_of_people_in_gallery_similar_enough)
        If the access is Accepted -> (True, [id_person#1, id_person#2, ...]
        If the access is Denied -> (False, None)
        """
        results_tests_probes = list()
        for probe_image in probes_images:
            results_tests_probes.append(self.authenticate(probe_image, gallery_images))
        return np.array(results_tests_probes)

    def compute_metrics(self, result_probes_authentication):
        """
        Computes metrics for a given results list of probes authentication
        :param result_probes_authentication: list of probes authentication
        :return: accuracy, precision, recall, specificity

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        """

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i, probe_authentication in enumerate(result_probes_authentication):
            # True positive : access authorized for a registered person
            if probe_authentication[0] and self.ground_truth[i][0] in probe_authentication[1]:
                TP += 1

            # False positive : access authorized for an unregistered person
            if probe_authentication[0] and self.ground_truth[i][0] not in probe_authentication[1]:
                FP += 1

            # True negative : access denied for an unregistered person
            if not(probe_authentication[0]) and not(self.ground_truth[i][0]):
                TN += 1

            # False negative : access denied for a registered person
            if not(probe_authentication[0]) and self.ground_truth[i][0]:
                FN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)

        return accuracy, precision, recall, specificity
