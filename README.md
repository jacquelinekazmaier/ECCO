# ECCO
Proof of concept implementation of the ECCO framework for Evaluating a Corpus Characterised by Opinion-bearing language

The repository is set up as follows:
- `preprocessing.py` contains functions for data handling, cleaning and preprocessing
- `modelling.py` contains functions for building, training and testing sentiment classification models
- `dashboard.py` contains functions for visualising and analysing model results by means of the Dashboard interface
- `mainwindow.ui` creates the PyQt GUI 
- `configurationApp.py` creates modern-looking version of configurations GUI, links functions to the GUI and runs the configurations app
- `saved_models`* is the directory in which models are saved by the system
- `log`* is the directory in which the results of comparative model evaluations are saved
- `logs`* is the directory in which TensorBoard files are saved for deep learning models
- `assets` and `graphics`* are used for the storage of internally generated images and files

\* These repositories will be created as needed when running the code

## Getting started
This code was built in `Python 3.7`. In order to be able to run the code, first install the required packages documented in `requirements.txt` by running the command

`pip install -r requirements.txt`.


## Running the code
1. Launch the _MainWindow_ interface by running `configurationApp.py`
2. Follow the instructions in the GUI to upload a _reviews_ data set and a _supplementary_ data set
3. Categorise the data attributes appropriately
4. Use the GUI to preprocess the text data
5. Use the GUI to select existing models or build custom models
6. Evaluate and compare developed models by means of the GUI
7. Select a model to analyse its results by means of the _Dashboard_ interface
8. Explore the model results and identify latent relationships between data attributes by means of the outputs of the _Dashboard_ interface

## Contact

Please address any queries to jqkazmaier@gmail.com
