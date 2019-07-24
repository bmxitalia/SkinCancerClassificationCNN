# Skin cancer classification on HAM10000
This repository contains the project made for the course of Cognitive Services at the Master degree in computer science at the University of Padua. 

In `report/` folder there is the paper associated to the project. It explains the choices, the approach and the dataset that we used in the project.

In `code/` folder there is the code to reproduce the experiments performed for the project and explained in the project report:

- `model_building.py` creates the five CNN models explained in the *Proposed model* section of the report;
- `model_testing.py` computes the metrics scores associated with the five models built by the previous script;
- `learning_rate_test.py` trains the proposed model with the three learning rate values explained in the *Optimizer and optimizer hyperparameters choice* section of the report;
- `data_augmentation.py` creates the oversampled dataset to perform the experiment explained in the *Experiments* section of the report;
- `model.py` performs all the experiments presented in the *Experiments* section of the report;
- `model_test.py` computes the metrics scores associated with the models built by the previous script.

In `results/` folder there are all the results obtained in this project and showed in the project report.

In *presentationSlides/* folder there are the slides used to present this project.

In *courseSlides/* folder there are the slides used by Professors to explain the course.

In *usefullInfo/* folder there are some useful informations about skin cancer classification task and deep learning topics.

