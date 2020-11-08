##Bastien DELFORGE
##Adrien MALGOWSKI 

# PROJECT ODATA : Facial authentication

**gallery_probes_generator.py** : object that creates a gallery and probes from pictures in /images folder
The created datasets will by saved in the folder /npy next to /images folder

**eigenface_generator.py** : object that computes and save the eigenfaces in the /eigen_faces_npy folder next to /images folder

**authenticator.py** : object that represents the facial authentication process and computes metrics

**radius_serach.py** : functions to computes distances with pictures of coefficients

**generate_all_npy** : pyton cript to generate ALL .npy usefull files for dataset1 and 2
-> gallery name and pictures
-> probes name and pictures
-> eigenface and mean face
-> gallery_pictures_eigenface and probes_pictures_eigenfaces

**Warning** : All the datasets are very long to compute

The jupyter notebooks are described well enough in our project report.