##Bastien DELFORGE
##Adrien MALGOWSKI 

# PROJECT ODATA : Facial authentication (without opencv)

!!! PLEASE !!!
place  dataset1 and dataset2 folder in a new folder called "data" 

the project needs to be like this :

Projet Bastien Adrien
|
|/data|
       |/datasets1|
                  |images/
                  |eigen_faces_npy/
                  |npy/
                  README.md
       |/datasets2|
                 |images/
                 |eigen_faces_npy/
                 |npy/
                 README.md
|authenticator.py
|gallery_probes_generator.py
|radius_serach.py
|README.md
|generate_all_npy.py
|eigenface_generator.py
|eigenface_notebook.ipynb
|test_parameters_impact.ipynb
|test_perf_authentificaton.ipnb


To generate npy/ and eigen_faces_npy/ folders and their content please run generate_all_npy.py
if you don't generate the folders and their contents, the project won't work.

**gallery_probes_generator.py** : object that creates a gallery and probes from pictures in /images folder
The created datasets will by saved in the folder /npy next to /images folder

**eigenface_generator.py** : object that computes and save the eigenfaces in the /eigen_faces_npy folder next to /images folder

**authenticator.py** : object that represents the facial authentication process and computes metrics

**radius_serach.py** : functions to computes distances with pictures and coefficients

**generate_all_npy.py** : python script to generate ALL .npy usefull files for dataset1 and 2
-> gallery name and pictures
-> probes name and pictures
-> eigenface and mean face
-> gallery_pictures_eigenface and probes_pictures_eigenfaces

**Warning** : All the datasets are very long to compute

The jupyter notebooks are described well enough in our project report.
