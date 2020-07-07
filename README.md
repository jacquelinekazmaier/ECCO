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
- `dictGB` contains English dictionaries necessary for spelling correction processes
- `data` houses the data set which maps South African town and province names to geographic coordinates as well as minimal dummy data sets to execute the example below

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
7. (optional) Combine several models into an ensemble:
    - Select several lines in the table
    - Click the "combine selected models in ensemble" button 
    - Select desired configurations
    - Click "OK" and return to 6
7. Select a model to analyse its results by means of the _Dashboard_ interface. Make sure the checkbox above the "save selected model" button is checked before clicking the button to proceed. 
8. Explore the model results and identify latent relationships between data attributes by means of the outputs of the _Dashboard_ interface

## Executing the example
1.  Launch the _MainWindow_ interface by running `configurationApp.py` 
2. Upload _example_reviews_data_set.csv_ as the Reviews Data and _example_supplementary_data_set.csv_ as the Supplementary data
3. Choose the following categorisation of the reviews data:
- review text: _ReviewText_
- sentiment label: _SentimentLabels_
- Location latitude or town name: _Town_
- Location longitude or province name: _Province_
- Location name: _Town_
- Date indicator: _Date_

Choose the following categorisation of the supplementary data:
- Link to review data: _ID_
- Corresponding link from review data: _ID_
- Qualitative attributes: _Genre_
- Quantitative attributes: _Running time_
4. Use the GUI to preprocess the text data as desired 
5. Use the GUI to select existing models or build custom models (NB: Ensure that the number of test and validation samples is greater than or equal to the number of folds used during cross-validation, e.g. change the test and validation data proportions to 30% for 3-fold CV)
6. Evaluate and compare developed models by means of the GUI
7. (Optional) Combine models in ensemble (only available if multiple custom models have been created):
- Select two or more rows in the table
- Click the "combine selected models in ensemble" button 
- Select desired configurations
- Click "OK"
8. Select a model to analyse its results by means of the _Dashboard_ interface. Make sure the checkbox above the "save selected model" button is checked before clicking the button to proceed. 
9. Explore the model results and identify latent relationships between data attributes by means of the outputs of the _Dashboard_ interface

## Contact

Please address any queries to jqkazmaier@gmail.com
