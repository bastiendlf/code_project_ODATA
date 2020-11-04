import matplotlib.image as mpimg
import numpy as np
import os
from random import sample


class GalleryProbesGenerator:

    def __init__(self, path_to_dataset='data/dataset1/', probes_length=200):
        self.path_to_dataset = path_to_dataset
        self.path_dataset_npy = path_to_dataset + '/npy'
        self.path_to_images = path_to_dataset + '/images'

        self.probes_length = probes_length

    def generate_npy_files(self):
        """

        :return:
        """
        self._generate_dataset_all_elements()
        self._generate_gallery_probes()

    def _generate_dataset_all_elements(self):
        """

        :return:
        """
        file_names = os.listdir(self.path_to_images)
        all_pictures_by_person = dict()

        for picture in file_names:
            current_name = picture.split(".")[0]

            if current_name not in all_pictures_by_person:
                all_pictures_by_person[current_name] = list()

            all_pictures_by_person[current_name].append(mpimg.imread(f'data/dataset1/images/{picture}'))

        for key, value in all_pictures_by_person.items():
            all_pictures_by_person[key] = np.array(value)  # converting python lists into numpy arrays

        names_list = np.array(list(all_pictures_by_person.keys()), dtype=object)
        pictures_list = np.array(list(all_pictures_by_person.values()), dtype=object)

        if os.path.exists(self.path_dataset_npy):
            import shutil
            shutil.rmtree(self.path_dataset_npy)  # remove file and its content to place new data in it

        os.makedirs(self.path_dataset_npy)
        np.save(f'{self.path_dataset_npy}/dataset_all_pictures.npy', pictures_list)
        np.save(f'{self.path_dataset_npy}/dataset_all_names.npy', names_list)

    def _generate_gallery_probes(self):
        """

        :return:
        """

        gallery_names = np.load(self.path_dataset_npy + "/dataset_all_names.npy", allow_pickle=True)
        gallery_pictures = np.load(self.path_dataset_npy + "/dataset_all_pictures.npy", allow_pickle=True)

        # copy for tests after splitting data
        dataset_all_names = np.load("data/dataset1/npy/dataset_all_names.npy", allow_pickle=True)
        dataset_all_pictures = np.load("data/dataset1/npy/dataset_all_pictures.npy", allow_pickle=True)

        probe_id_global = sample(range(0, gallery_names.shape[0] - 1), self.probes_length)
        probe_id_registered = probe_id_global[:int(self.probes_length / 2)]
        probe_id_unregistered = probe_id_global[int(self.probes_length / 2):]

        # Creating probes for registered people
        probe_registered_names = list()
        probe_registered_pictures = list()
        pictures_to_delete_from_gallery = list()  # will be useful later to check if probe_register_pictures correctly created

        for probe_id in probe_id_registered:
            probe_registered_names.append(gallery_names[probe_id])
            # choosing a random picture in the array of pictures
            picture_id = sample(range(0, len(gallery_pictures[probe_id]) - 1), 1)[0]
            pictures_to_delete_from_gallery.append((probe_id, picture_id))

            probe_registered_pictures.append(gallery_pictures[probe_id][picture_id])
            gallery_pictures[probe_id] = np.delete(gallery_pictures[probe_id], [picture_id], axis=0)

        # converting python lists into numpy arrays
        probe_registered_names = np.array(probe_registered_names)
        probe_registered_pictures = np.array(probe_registered_pictures)

        # test to see if probe_register names and pictures well created
        for i, (current_id, picture_id) in enumerate(pictures_to_delete_from_gallery):

            # check if one picture is correctly deleted for the current person in the gallery
            if len(gallery_pictures[current_id]) + 1 != len(dataset_all_pictures[current_id]):
                print(f"{gallery_names[current_id]} -> wrong pictures number after putting in probe registered")

            # check if the right picture is placed in probe_registered_pictures
            if not (np.array_equal(probe_registered_pictures[i], dataset_all_pictures[current_id][picture_id])):
                print(f'{probe_registered_names[i]} -> wrong picture copied')

            # check if the picture placed in probe_registered_pictures is the one deleted
            if np.array_equal(gallery_pictures[current_id][picture_id], dataset_all_pictures[current_id][picture_id]):
                print(f"{probe_registered_names[i]} -> wrong picture deleted in gallery")

        # Creating probes for unregistered people
        probe_unregistered_names = list()
        probe_unregistered_pictures = list()
        people_to_delete_from_gallery = list()  # will be useful later to check if probe_register_pictures correctly created

        for probe_id in probe_id_unregistered:
            probe_unregistered_names.append(gallery_names[probe_id])
            # choosing a random picture in the array of pictures
            picture_id = sample(range(0, len(gallery_pictures[probe_id]) - 1), 1)[0]
            people_to_delete_from_gallery.append((probe_id, gallery_names[probe_id]))

            probe_unregistered_pictures.append(gallery_pictures[probe_id][picture_id])

        # converting python lists into numpy arrays
        probe_unregistered_names = np.array(probe_unregistered_names)
        probe_unregistered_pictures = np.array(probe_unregistered_pictures)

        # removing the unregistered elements and their pictures
        gallery_names = np.delete(gallery_names, probe_id_unregistered)
        gallery_pictures = np.delete(gallery_pictures, probe_id_unregistered)

        # test to see if probe_unregister names and pictures well created
        for i, (current_id, current_name) in enumerate(people_to_delete_from_gallery):

            # check if all the names are removed
            if current_name in gallery_names:
                print(f'{current_name} not deleted from gallery')
            # check if all probes correctly added
            if current_name not in probe_unregistered_names:
                print(f'{current_name} not in probes unregister')

            if gallery_pictures.shape[0] != dataset_all_pictures.shape[0] - self.probes_length / 2:
                print(f'Not the correct number of element in pictures gallery')

        # saving probes and gallery offline
        # gallery
        np.save(self.path_dataset_npy + '/gallery_names.npy', gallery_names)
        np.save(self.path_dataset_npy + '/gallery_pictures.npy', gallery_pictures)

        # registered probes
        np.save(self.path_dataset_npy + '/probe_registered_names.npy', probe_registered_names)
        np.save(self.path_dataset_npy + '/probe_registered_pictures.npy', probe_registered_pictures)

        # unregistered probes
        np.save(self.path_dataset_npy + '/probe_unregistered_names.npy', probe_unregistered_names)
        np.save(self.path_dataset_npy + '/probe_unregistered_pictures.npy', probe_unregistered_pictures)

    def get_gallery(self):
        """

        :return:
        """
        gallery_names = np.load(self.path_dataset_npy + '/gallery_names.npy', allow_pickle=True)
        gallery_pictures = np.load(self.path_dataset_npy + '/gallery_pictures.npy', allow_pickle=True)

        return gallery_names, gallery_pictures

    def get_registered_probes(self):
        """

        :return:
        """
        probe_registered_names = np.load(self.path_dataset_npy + '/probe_registered_names.npy', allow_pickle=True)
        probe_registered_pictures = np.load(self.path_dataset_npy + '/probe_registered_pictures.npy', allow_pickle=True)

        return probe_registered_names, probe_registered_pictures

    def get_unregistered_probes(self):
        """

        :return:
        """
        probe_unregistered_names = np.load(self.path_dataset_npy + '/probe_unregistered_names.npy', allow_pickle=True)
        probe_unregistered_pictures = np.load(self.path_dataset_npy + '/probe_unregistered_pictures.npy',
                                              allow_pickle=True)

        return probe_unregistered_names, probe_unregistered_pictures
