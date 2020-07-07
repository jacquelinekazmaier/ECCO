'''Creates modern-looking version of configurations GUI, links functions to GUI and runs the configurations app'''

import sys
import random
import matplotlib as mpl
mpl.use('QT5Agg')

import qtmodern.styles
import qtmodern.windows
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

from os.path import join, dirname, abspath
from qtpy import uic, QtGui
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QDialog
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import DenseTransformer
from joblib import dump, load
from datetime import datetime
from collections import Counter

'''Model'''
import modelling as modelling
from preprocessing import ReviewData, ClientData
from dashboard import Dashboard

punctuation = [c for c in string.punctuation]

'''View'''
_UI = join(dirname(abspath(__file__)), 'mainwindow.ui')
_UI2 = join(dirname(abspath(__file__)), 'ensemble.ui')

'''Controller'''
def get_feature_settings(self):
    document_models = []
    if self.presenceBox.isChecked():
        document_models.append('presence')
    if self.frequencyBox.isChecked():
        document_models.append('frequency')
    if self.tfidfBox.isChecked():
        document_models.append('tf-idf')

    ngram_ranges = []
    if self.unigramsBox.isChecked():
        ngram_ranges.append((1, 1))
    if self.bigramsBox.isChecked():
        ngram_ranges.append((2, 2))
    if self.trigramsBox.isChecked():
        ngram_ranges.append((3, 3))
    if self.unibigramsBox.isChecked():
        ngram_ranges.append((1, 2))
    if self.bitrigramsBox.isChecked():
        ngram_ranges.append((2, 3))
    if self.unibitrigramsBox.isChecked():
        ngram_ranges.append((1, 3))

    test_data = float(self.testTxt.text()) / 100
    val_data = float(self.valTxt.text()) / 100
    val_data = 1 - (1-val_data-test_data)/(1 - test_data) #express as % of training data

    n_folds = self.cvfoldsspinBox.value()
    metric = self.metricDdl.currentText()
    maxfeat = self.maxFeaturexBox.value()

    return document_models, ngram_ranges, test_data, val_data, n_folds, metric, maxfeat

class ModelClass:
    def __init__(self):
        self.model = BaseEstimator()
        self.crossval_score = 0
        self.param_grid = {}
        self.best_params = {}

class Experiment:
    def __init__(self, algorithm, modelclass, document_representation, ngram_range, sampling_method, maxfeat):
        self.modelclass = modelclass
        self.algorithm = algorithm
        self.document_representation = document_representation
        self.ngram_range = ngram_range
        self.sampling_method = sampling_method
        self.max_features = maxfeat

    def log_results(self, auc, acc, prec, rec, f1):
        self.auc = auc
        self.accuracy = acc
        self.f1 = f1
        self.precision = prec
        self.recall = rec

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi(_UI, self)
        self.labelsBox.stateChanged.connect(self.on_labelsBox_stateChanged)
        self.stopwordsBox.stateChanged.connect(self.on_stopwordsBox_stateChanged)
        self.punctuationBox.stateChanged.connect(self.on_punctuationBox_stateChanged)
        self.groupnumBox.stateChanged.connect(self.on_groupnumBox_stateChanged)
        self.nbBox.stateChanged.connect(self.on_nbBox_stateChanged)
        self.svmBox.stateChanged.connect(self.on_svmBox_stateChanged)
        self.logregBox.stateChanged.connect(self.on_logregBox_stateChanged)
        self.annBox.stateChanged.connect(self.on_annBox_stateChanged)
        self.cnnBox.stateChanged.connect(self.on_cnnBox_stateChanged)
        self.lstmBox.stateChanged.connect(self.on_lstmBox_stateChanged)
        self.comparetable.itemSelectionChanged.connect(self.on_comparetable_itemSelectionChanged)
        self.testTxt.textChanged.connect(self.on_testTxt_textChanged)
        self.valTxt.textChanged.connect(self.on_valTxt_textChanged)
        self.stack.widget(0).setStyleSheet(".QWidget{border-image: url(assets/eccoEnsembles.png)};")
        self.selected_seed_index = 0

        class EnsembleWindow(QDialog):
            def __init__(self):
                QDialog.__init__(self)
                uic.loadUi(_UI2, self)
                self.metaradioButton.toggled.connect(self.on_metaradioButton_StateChanged)
                self.configuration = {}
                self.result = 0

            @Slot()
            def on_metaradioButton_StateChanged(self):
                if self.metaradioButton.isChecked():
                    self.groupBox_2.setEnabled(True)
                else:
                    self.groupBox_2.setEnabled(False)

            @Slot()
            def accept(self):
                if self.scoringBtn.isChecked():
                    self.configuration['outputType'] = 'score'
                else:
                    self.configuration['outputType'] = 'discrete'

                if self.votingradioButton.isChecked():
                    self.configuration['combinationMethod'] = 'simple'
                elif self.weightedradioButton.isChecked():
                    self.configuration['combinationMethod'] = 'weighted'
                else:
                    self.configuration['combinationMethod'] = 'meta'
                    self.configuration['meta-learner'] = self.metalearnerDdl.currentText()

                self.result = 1
                self.close()

        self.ew = EnsembleWindow()

    ################################################
    # Functions for home page
    ################################################
    @Slot()
    def on_startButton_clicked(self):
        self.stack.setCurrentIndex(1)

    ################################################
    # Functions for data upload page
    ################################################
    @Slot()
    def on_uploadButton_clicked(self):
        # get data and create reviews instance
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*)")
        self.reviews = ReviewData(filename)  # create reviews class instance and store for use inside Window

        # display filename
        self.dataFilename.setText("Current file: " + filename.rsplit('/', 1)[-1])

        # populate dropdownlists
        self.textfieldDdl.clear()
        self.textfieldDdl.addItems(self.reviews.data.columns)
        self.labelfieldDdl.clear()
        self.labelfieldDdl.addItems(self.reviews.data.columns)

        self.dateDdl.clear()
        self.dateDdl.addItems(['None'])
        self.dateDdl.addItems(self.reviews.data.columns)

        self.latDdl.clear()
        self.latDdl.addItems(['None'])
        self.latDdl.addItems(self.reviews.data.columns)
        self.longDdl.clear()
        self.longDdl.addItems(['None'])
        self.longDdl.addItems(self.reviews.data.columns)
        self.locDdl.clear()
        self.locDdl.addItems(['None'])
        self.locDdl.addItems(self.reviews.data.columns)

        # populate list boxes
        self.qualiReviewDetailsList.clear()
        self.qualiReviewDetailsList.addItems(self.reviews.data.columns)
        self.quantiReviewDetailsList.clear()
        self.quantiReviewDetailsList.addItems(self.reviews.data.columns)
        self.clientuploadButton.setEnabled(True)

    @Slot()
    def on_labelsBox_stateChanged(self):
        if self.labelsBox.isChecked():
            self.labelfieldDdl.setEnabled(True)
            self.labelddlLabel.setEnabled(True)
        else:
            self.labelfieldDdl.setEnabled(False)
            self.labelddlLabel.setEnabled(False)

    @Slot()
    def on_clientuploadButton_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*)")
        self.clients = ClientData(filename)  # create client data class instance and store for use inside Window

        # display filename
        self.clientdataFilename.setText("Current file: " + filename.rsplit('/', 1)[-1])

        # populate dropdownlists
        self.idInClientDataDdl.clear()
        self.idInClientDataDdl.addItems(self.clients.df.columns)
        self.idInReviewDataDdl.clear()
        self.idInReviewDataDdl.addItems(self.reviews.data.columns)

        # populate list boxes
        self.qualiCustomerDetailsList.clear()
        self.qualiCustomerDetailsList.addItems(self.clients.df.columns)
        self.quantiCustomerDetailsList.clear()
        self.quantiCustomerDetailsList.addItems(self.clients.df.columns)

    @Slot()
    def on_loadDataBtn_clicked(self):
        self.stack.setCurrentIndex(2)

        # store text and labels
        print('storing review data . . . ')
        self.reviews.store_text(str(self.textfieldDdl.currentText()))
        self.reviews.store_additional_fields(locations=[self.latDdl.currentText(), self.longDdl.currentText()],
                                             qual_profiles=[item.text() for item in
                                                            self.qualiReviewDetailsList.selectedItems()],
                                             quant_profiles=[item.text() for item in
                                                             self.quantiReviewDetailsList.selectedItems()],
                                             date=self.dateDdl.currentText(),
                                             loc=self.locDdl.currentText()
                                             )
        if self.labelfieldDdl.isEnabled():
            self.reviews.process_labels(str(self.labelfieldDdl.currentText()))

        # store customer data
        print('storing supplementary data . . . ')
        quali = [item.text() for item in self.qualiReviewDetailsList.selectedItems()]
        quali.extend([item.text() for item in self.qualiCustomerDetailsList.selectedItems()])

        quanti = [item.text() for item in self.quantiReviewDetailsList.selectedItems()]
        quanti.extend([item.text() for item in self.quantiCustomerDetailsList.selectedItems()])

        try:
            self.clients.save_relevant_fields(links={'clients_side': self.idInClientDataDdl.currentText(),
                                                     'reviews_side': self.idInReviewDataDdl.currentText()},
                                              qual_profiles=[item.text() for item in
                                                             self.qualiCustomerDetailsList.selectedItems()],
                                              quant_profiles=[item.text() for item in
                                                              self.quantiCustomerDetailsList.selectedItems()]
                                              )
            self.clients.clean(threshold=0.25)
        except:
            pass

        # create and display wordcloud
        all_reviews = " ".join(review for review in self.reviews.text).split(" ")
        most_common = {word: word_count for word, word_count in Counter(all_reviews).most_common(1000)}
        print('creating word cloud . . . ')
        path = self.reviews.create_wordcloud(most_common)
        image_profile = QtGui.QImage(path)  # QImage object
        self.wordcloud.setPixmap(QtGui.QPixmap.fromImage(image_profile))

    ################################################
    # Functions for data preprocessing page
    ################################################
    @Slot()
    def on_continuetomodelButton_clicked(self):
        test_num = "Amounts to {} observations".format(round(len(self.reviews.labels) * 20 / 100), 0)
        val_num = "Amounts to {} observations".format(round(len(self.reviews.labels) * 10 / 100), 0)
        self.testNumLabel.setText(test_num)
        self.valNumLabel.setText(val_num)
        self.seed = random.randint(0, 200)
        self.num_clicks = 0
        self.stack.setCurrentIndex(4)

    @Slot()
    def on_backtouploadButton_clicked(self):
        self.stack.setCurrentIndex(1)

    @Slot()
    def on_labelsBox_stateChanged(self):
        if self.labelsBox.isChecked():
            self.labelfieldDdl.setEnabled(True)
            self.labelddlLabel.setEnabled(True)
        else:
            self.labelfieldDdl.setEnabled(False)
            self.labelddlLabel.setEnabled(False)

    @Slot()
    def on_stopwordsBox_stateChanged(self):
        if self.stopwordsBox.isChecked():
            self.stopwordsList.setEnabled(True)
            self.stopwordsList.addItems(stopwords.words('english'))
        elif self.punctuationBox.isChecked() == False:
            self.stopwordsList.clear()
            self.stopwordsList.setEnabled(False)
        else:
            self.stopwordsList.clear()
            self.stopwordsList.addItems([c for c in string.punctuation])

    @Slot()
    def on_punctuationBox_stateChanged(self):
        if self.punctuationBox.isChecked():
            self.stopwordsList.setEnabled(True)
            self.stopwordsList.addItems([c for c in string.punctuation])
        elif self.stopwordsBox.isChecked() == False:
            self.stopwordsList.clear()
            self.stopwordsList.setEnabled(False)
        else:
            self.stopwordsList.clear()
            self.stopwordsList.addItems(stopwords.words('english'))

    @Slot()
    def on_groupnumBox_stateChanged(self):
        if self.groupnumBox.isChecked():
            self.numbersExcludeLabel.setEnabled(True)
            self.numbersExcludeTxt.setEnabled(True)
        else:
            self.numbersExcludeLabel.setEnabled(False)
            self.numbersExcludeTxt.setEnabled(False)

    @Slot()
    def on_preprocessingBtn_clicked(self):
        'Apply selected preprocessing steps to the data'

        # tokenise
        print('tokenising...')
        self.preprocessingProgress.setValue(0)
        self.reviews.tokenise()
        self.preprocessingProgress.setValue(30)

        self.reviews.get_types()
        original_num_tokens = str(len(self.reviews.types))

        # stopword and punctuation removal
        print('removing stopwords and punctuation...')
        if self.stopwordsList.isEnabled():
            stop_words = []
            for x in range(self.stopwordsList.count() - 1):
                if self.stopwordsList.item(x) not in self.stopwordsList.selectedItems():
                    stop_words.append(self.stopwordsList.item(x).text())
            self.reviews.remove_stopwords(stop_words)
            self.preprocessingProgress.setValue(50)

        # spell checking
        print('correcting spelling...')
        if self.spellcheckBox.isChecked():
            if self.groupnumBox.isChecked():
                self.reviews.spellcheck('aspell', num=True, exclude_list=self.numbersExcludeTxt.toPlainText())
            else:
                self.reviews.spellcheck('aspell', num=False)
        elif self.groupnumBox.isChecked():
            print(self.numbersExcludeTxt.toPlainText())
            self.reviews.spellcheck('None', num=True, exclude_list=self.numbersExcludeTxt.toPlainText())

        # make all lowercase
        print('lowercasing...')
        if self.lowercaseBox.isChecked():
            self.reviews.make_lowercase()

        # stemming or lemmatisation
        print('stemming...')
        if self.porterRbtn.isChecked():
            self.reviews.stem('Porter')
        elif self.lancasterRbtn.isChecked():
            self.reviews.stem('Lancaster')
        elif self.snowballRbtn.isChecked():
            self.reviews.stem('Snowball')
        elif self.lemmatisationRbtn.isChecked():
            self.reviews.stem('Lemmatisation')

        try:
            print('spell corrected:')
            print(self.reviews.replaced)
        except:
            pass

        # store as text for vectoriser later
        print('storing as text...')
        self.reviews.store_as_text()
        
        # get unique tokens (types) of the corpus
        self.reviews.get_types()

        # create new wordcloud
        self.preprocessingProgress.setValue(80)
        print('creating word cloud . . . ')
        path = self.reviews.create_wordcloud(self.reviews.tokens_dist)
        image_profile = QtGui.QImage(path)  # QImage object
        self.wordcloud.setPixmap(QtGui.QPixmap.fromImage(image_profile))
        self.preprocessingProgress.setValue(95)

        # get unique tokens (types) of the corpus
        self.numreviewsLabel.setText(str(len(self.reviews.text)))
        self.numtypesLabel.setText(original_num_tokens)
        self.numtypesafterLabel.setText(str(len(self.reviews.types)))
        self.preprocessingProgress.setValue(100)

        self.savedModelButton.setEnabled(True)
        if self.labelfieldDdl.isEnabled():
            self.continuetomodelButton.setEnabled(True)

    @Slot()
    def on_savedModelButton_clicked(self):
        self.stack.setCurrentIndex(3)

    ################################################
    # Functions for saved model page
    ################################################
    @Slot()
    def on_backtopreprocessing2Button_clicked(self):
        self.stack.setCurrentIndex(2)
    @Slot()
    def on_selectModelBtn_clicked(self):
        try:
            if len(set(self.reviews.labels)) == 2:
                binary=True
            else:
                binary=False
            classes = np.unique(self.reviews.labels)
        except:
            binary = False
            classes = np.unique(['positive', 'negative', 'neutral'])

        if self.sentiRBtn.isChecked():
            swn = modelling.Sentiwordnet()
            swn.classes_ = classes
            self.finalpredictions = swn.predict(self.reviews.cleantext, binary=binary)
        if self.patternRBtn.isChecked():
            ptn = modelling.Pattern_sentiment()
            ptn.classes_ = classes
            self.finalpredictions = ptn.predict(self.reviews.cleantext, binary=binary)
        if self.huLiuRBtn.isChecked():
            hl = modelling.Hu_liu_sentiment()
            hl.classes_ = classes
            self.finalpredictions = hl.predict(self.reviews.cleantext, binary=binary)
        if self.vaderRBtn.isChecked():
            vdr = modelling.Vader()
            vdr.classes_ = classes
            self.finalpredictions = vdr.predict(self.reviews.cleantext, binary=binary)
        if self.savedRBtn.isChecked():
            filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*)")
            try:
                model = load(filename)
                self.finalpredictions = model.predict(self.reviews.cleantext)
            except:
                pass
        try:
            if np.array(self.finalpredictions)[0].shape[0]==1: #if funny 2D output of NN model
                self.finalpredictions = [pred[0] for pred in self.finalpredictions]
        except:
            pass

        # Launch Dash App
        print("Deploying model...")
        try:
            final = pd.DataFrame({'Client_num': self.reviews.data[self.clients.links['reviews_side']],
                                  'Review': self.reviews.text,
                                  'Sentiment': self.finalpredictions},
                                 )
            final = pd.concat([final, self.reviews.df], axis=1)
            my_dash = Dashboard(final_predictions=final, reviews=self.reviews, customer_data=self.clients)
        except:
            final = pd.DataFrame({'Client_num': None,
                                  'Review': self.reviews.text,
                                  'Sentiment': self.finalpredictions},
                                 )
            final = pd.concat([final, self.reviews.df], axis=1)
            my_dash = Dashboard(final_predictions=final, reviews=self.reviews, customer_data=None)

        print("Closing window...")
        self.close()
        print("Launching Dash App...")
        my_dash.deploy()

    ################################################
    # Functions for modelling page
    ################################################
    @Slot()
    def on_backtoPreprocessingButton_clicked(self):
        self.stack.setCurrentIndex(2)

    @Slot()
    def on_testTxt_textChanged(self):
        test = int(self.testTxt.text())
        test_num = "Amounts to {} observations".format(round(len(self.reviews.labels)*test/100), 0)
        self.testNumLabel.setText(test_num)

    @Slot()
    def on_valTxt_textChanged(self):
        val = int(self.valTxt.text())
        val_num = "Amounts to {} observations".format(round(len(self.reviews.labels) * val / 100), 0)
        self.valNumLabel.setText(val_num)

    @Slot()
    def on_nbBox_stateChanged(self):
        if self.nbBox.isChecked():
            self.nbFrame.setEnabled(True)
        else:
            self.nbFrame.setEnabled(False)

    @Slot()
    def on_svmBox_stateChanged(self):
        if self.svmBox.isChecked():
            self.svmFrame.setEnabled(True)
        else:
            self.svmFrame.setEnabled(False)

    @Slot()
    def on_logregBox_stateChanged(self):
        if self.logregBox.isChecked():
            self.logregFrame.setEnabled(True)
        else:
            self.logregFrame.setEnabled(False)

    @Slot()
    def on_annBox_stateChanged(self):
        if self.annBox.isChecked():
            self.annFrame.setEnabled(True)
        else:
            self.annFrame.setEnabled(False)

    @Slot()
    def on_cnnBox_stateChanged(self):
        if self.cnnBox.isChecked():
            self.cnnFrame.setEnabled(True)
        else:
            self.cnnFrame.setEnabled(False)

    @Slot()
    def on_lstmBox_stateChanged(self):
        if self.lstmBox.isChecked():
            self.lstmFrame.setEnabled(True)
        else:
            self.lstmFrame.setEnabled(False)

    @Slot()
    def on_annTBbutton_clicked(self):
        self.ANNthread = modelling.LaunchTensorboard(model='ANN')
        self.ANNthread.start()

    @Slot()
    def on_cnnTBbutton_clicked(self):
        self.CNNthread = modelling.LaunchTensorboard(model='CNN')
        self.CNNthread.start()

    @Slot()
    def on_lstmTBbutton_clicked(self):
        self.LSTMthread = modelling.LaunchTensorboard(model='LSTM')
        self.LSTMthread.start()

    @Slot()
    def on_annTestButton_clicked(self):
        help_string = [i.split(",") for i in self.annHList.text().split(";")][0]
        hidden_layers = [int(k) for k in help_string]

        if self.ANNreluBox.isChecked():
            activation_fct = 'relu'
        elif self.ANNsigmoidBox.isChecked():
            activation_fct='sigmoid'
        elif self.ANNtanhBox.isChecked():
            activation_fct='tanh'

        if self.ANNhingeBox.isChecked():
            loss_fct = 'hinge'
        elif self.ANNentropyBox.isChecked():
            loss_fct = 'binary_crossentropy'

        if self.ANNL1Box.isChecked():
            reg_type = 'L1'
            help_string = [i for i in self.annLambdaList.text().split(",")][0]
            reg_param = float(help_string)
        elif self.ANNL2Box.isChecked():
            reg_type = 'L2'
            help_string = [i for i in self.annLambdaList.text().split(",")][0]
            reg_param = float(help_string)
        else:
            reg_type=None
            reg_param=0

        dropout_prob = [float(c) for c in self.annDropoutList.text().split(",")][0]

        if self.ANNbnyesBox.isChecked():
            batchnorm = True
        elif self.ANNbnnoBox.isChecked():
            batchnorm = False

        if self.ANNsvgBox.isChecked():
            solver='svg'
        elif self.ANNmomentumBox.isChecked():
            solver='momentum'
        elif self.ANNadamBox.isChecked():
            solver='Adam'
        elif self.ANNrmsBox.isChecked():
            solver='RMSprop'

        max_epochs = [int(c) for c in self.annEpochsList.text().split(",")][0]

        lr = [float(c) for c in self.annLrList.text().split(",")][0]

        lr_decay = [float(i) for i in self.annLambdaList.text().split(",")][0]

        document_models, ngram_ranges, test_data, val_data, n_folds, metric, maxfeat = get_feature_settings(self)

        X_train, X_test, idx_train, idx_test, y_train, y_test = train_test_split(
            self.reviews.cleantext[self.reviews.labelled_indices],
            self.reviews.labelled_data.index, self.reviews.labels, test_size=test_data,
            random_state=self.seed, stratify=self.reviews.labels)

        ANN_test_model = modelling.ANN(hidden_layers=hidden_layers, activation_function=activation_fct, loss_function=loss_fct,
                    reg_type=reg_type, reg_param=reg_param, dropout_prob=dropout_prob, batchnorm=batchnorm,
                    solver=solver, max_epochs=max_epochs, learning_rate=lr, decay=lr_decay, val_split=val_data)
        vectoriser = modelling.vectorise(document_models[0], maxfeat, ngram_ranges[0])

        pipeline = Pipeline(steps=[('vectorise', vectoriser),
                                   ('to_dense', DenseTransformer()),
                                   ('model', ANN_test_model)])
        pipeline.fit(X_train, y_train)

    @Slot()
    def on_cnnTestButton_clicked(self):
        embedding_sizes = [int(i) for i in self.cnnEList.text().split(",")][0]

        help = [experiment for experiment in self.cnnFList.text().split(";")][0]
        experiment2 = []
        for layer in help.split(","):
            layer = layer[1:-1]  # remove brackets
            experiment2.append([int(k) for k in layer.split(" ")])
        hidden_layers = experiment2

        if self.cnnConvSameBox.isChecked():
            convolution_types = 'same'
        if self.cnnConvValidBox.isChecked():
            convolution_types = 'valid' #default valid

        if self.cnnPoolYesBox.isChecked():
            pooling ='yes'
        if self.cnnPoolNoBox.isChecked():
            pooling = 'no' #default no

        pooling_filter_sizes = [int(i) for i in self.cnnPoolFList.text().split(",")][0]
        pooling_filter_strides = [int(i) for i in self.cnnPoolSList.text().split(",")][0]

        if self.CNNreluBox.isChecked():
            activation_fct = 'relu' #default relu
        elif self.CNNsigmoidBox.isChecked():
            activation_fct = 'sigmoid'
        elif self.CNNtanhBox.isChecked():
            activation_fct = 'tanh'

        if self.CNNhingeBox.isChecked():
            loss_fct = 'hinge'
        if self.CNNentropyBox.isChecked():
            loss_fct = 'binary_crossentropy' #is default

        if self.CNNL1Box.isChecked():
            reg_type = 'L1'
        if self.CNNL2Box.isChecked():
            reg_type = 'L2'
        if self.CNNnoneBox.isChecked():
            reg_type = 'None' #default = none

        if self.CNNbnnoBox.isChecked():
            batchnorm = False
        elif self.CNNbnyesBox.isChecked():
            batchnorm = True
        else:
            batchnorm = False

        reg_param = [i for i in self.cnnLambdaList.text().split(",")][0]

        if self.CNNadamBox.isChecked():
            solver = 'Adam'
        elif self.CNNsvgBox.isChecked():
            solver = 'svg'
        elif self.CNNmomentumBox.isChecked():
            solver = 'momentum'
        elif self.CNNrmsBox.isChecked():
            solver = 'RMSprop'

        max_epochs = [int(c) for c in self.cnnEpochsList.text().split(",")][0]

        lr = [float(c) for c in self.cnnLrList.text().split(",")][0]

        lr_decay = [float(i) for i in self.cnnLambdaList.text().split(",")][0]

        document_models, ngram_ranges, test_data, val_data, n_folds, metric, maxfeat = get_feature_settings(
            self)

        X_train, X_test, idx_train, idx_test, y_train, y_test = train_test_split(
            self.reviews.cleantext[self.reviews.labelled_indices],
            self.reviews.labelled_data.index, self.reviews.labels, test_size=test_data,
            random_state=self.seed, stratify=self.reviews.labels)

        CNN_test_model = modelling.CNN(embedding_size = embedding_sizes, max_feat = maxfeat,
                                       hidden_layers=hidden_layers,
                                       convolution_type = convolution_types, pooling = pooling,
                                       pooling_filter_size = pooling_filter_sizes,
                                       pooling_filter_stride = pooling_filter_strides,
                                       activation_function=activation_fct,
                                       loss_function=loss_fct,
                                       reg_type=reg_type, reg_param=reg_param, batchnorm=batchnorm,
                                       solver=solver, max_epochs=max_epochs, learning_rate=lr, decay=lr_decay,
                                       val_split=val_data)
        CNN_test_model.fit(X_train, y_train)

    @Slot()
    def on_lstmTestButton_clicked(self):
        embedding_sizes = [int(i) for i in self.lstmEList.text().split(",")][0]

        help_string = [i.split(",") for i in self.lstmHList.text().split(";")][0]
        hidden_layers = [int(k) for k in help_string]

        dropout_prob = [float(c) for c in self.annDropoutList.text().split(",")][0]

        if self.lstmhingeBox.isChecked():
            loss_fct = 'hinge'
        elif self.lstmentropyBox.isChecked():
            loss_fct = 'binary_crossentropy'

        if self.lstmL1Box.isChecked():
            reg_type = 'L1'
        if self.lstmL2Box.isChecked():
            reg_type = 'L2'
        if self.lstmnoneBox.isChecked():
            reg_type = 'None'

        reg_param = [i for i in self.lstmLambdaList.text().split(",")][0]

        if self.lstmadamBox.isChecked():
            solver = 'Adam'
        elif self.lstmsvgBox.isChecked():
            solver = 'svg'
        elif self.lstmmomentumBox.isChecked():
            solver = 'momentum'
        elif self.lstmrmsBox.isChecked():
            solver = 'RMSprop'

        max_epochs = [int(c) for c in self.lstmEpochsList.text().split(",")][0]

        lr = [float(c) for c in self.lstmLrList.text().split(",")][0]

        lr_decay = [float(i) for i in self.lstmLambdaList.text().split(",")][0]

        document_models, ngram_ranges, test_data, val_data, n_folds, metric, maxfeat = get_feature_settings(
            self)

        X_train, X_test, idx_train, idx_test, y_train, y_test = train_test_split(
            self.reviews.cleantext[self.reviews.labelled_indices],
            self.reviews.labelled_data.index, self.reviews.labels, test_size=test_data,
            random_state=self.seed, stratify=self.reviews.labels)

        LSTM_test_model = modelling.LSTM(embedding_size=embedding_sizes, max_feat=maxfeat,
                                       hidden_layers=hidden_layers,
                                       dropout_prob = dropout_prob,
                                       loss_function=loss_fct,
                                       reg_type=reg_type, reg_param=reg_param,
                                       solver=solver, max_epochs=max_epochs, learning_rate=lr, decay=lr_decay,
                                       val_split=val_data)
        LSTM_test_model.fit(X_train, y_train)

    @Slot()
    def on_evaluateButton_clicked(self):
        try:
            self.ANNthread.terminate()
            print('stopped')
        except:
            print('not stopped')
        try:
            self.CNNthread.terminate()
            print('stopped')
        except:
            print('not stopped')
        try:
            self.LSTMthread.terminate()
            print('stopped')
        except:
            print('not stopped')
        tf.reset_default_graph()

        # configure settings for experiments
        document_models, ngram_ranges, test_data, val_data, n_folds, metric, maxfeat = get_feature_settings(self)

        # get selected models and parameters for cross-validation
        models = {}
        if self.nbBox.isChecked():
            models['NB'] = ModelClass()
            alpha = [float(a) for a in self.nbalphaTxt.text().split(",")]
            nb_params = {'model__alpha': alpha}
            models['NB'].param_grid = nb_params

        if self.svmBox.isChecked():
            models['SVM'] = ModelClass()
            svm_params = []
            svm_C = [float(c) for c in self.svmCList.text().split(",")]
            gamma = [float(g) for g in self.svmGammaList.text().split(",")]
            if self.svmKernelLinearBox.isChecked():
                svm_params.append({'model__kernel': ['linear'], 'model__C': svm_C})
            if self.svmKernelP1Box.isChecked() or self.svmKernelP2Box.isChecked() or self.svmKernelP1Box.isChecked():
                degree = []
                if self.svmKernelP1Box.isChecked():
                    degree.append(1)
                if self.svmKernelP2Box.isChecked():
                    degree.append(2)
                if self.svmKernelP3Box.isChecked():
                    degree.append(3)
                svm_params.append({'model__kernel': ['poly'], 'model__gamma': gamma, 'model__C': svm_C,
                                   'model__degree': degree})
            if self.svmKernelRadialBox.isChecked() or self.svmKernelSigmoidBox.isChecked():
                kernels = []
                if self.svmKernelRadialBox.isChecked():
                    kernels.append('rbf')
                if self.svmKernelSigmoidBox.isChecked():
                    kernels.append('sigmoid')
                svm_params.append({'model__gamma': gamma, 'model__C': svm_C, 'model__kernel': kernels})
            models['SVM'].param_grid = svm_params

        if self.logregBox.isChecked():
            models['LogReg'] = ModelClass()
            lg_C = [float(c) for c in self.logregCList.text().split(",")]
            solvers = []
            if self.logregSolverSAGBox.isChecked():
                solvers.append('sag')
            if self.logregSolverNewtonBox.isChecked():
                solvers.append('newton-cg')
            if self.logregSolverLBFGSBox.isChecked():
                solvers.append('lbfgs')
            models['LogReg'].param_grid = {'model__C': lg_C, 'model__solver': solvers, 'model__multi_class': ['auto'],
                                           'model__max_iter': [self.maxiterspinBox.value()], 'model__n_jobs': [8]}

        if self.annBox.isChecked():
            models['ANN'] = ModelClass()

            hidden_layers = []
            help = [i.split(",") for i in self.annHList.text().split(";")]
            for string in help:
                hidden_layers.append([int(k) for k in string])

            activation_fcts = []
            if self.ANNreluBox.isChecked():
                activation_fcts.append('relu')
            if self.ANNsigmoidBox.isChecked():
                activation_fcts.append('sigmoid')
            if self.ANNtanhBox.isChecked():
                activation_fcts.append('tanh')

            loss_fcts = []
            if self.ANNhingeBox.isChecked():
                loss_fcts.append('hinge')
            if self.ANNentropyBox.isChecked():
                loss_fcts.append('binary_crossentropy')

            reg_types = []
            if self.ANNL1Box.isChecked():
                reg_types.append('L1')
            if self.ANNL2Box.isChecked():
                reg_types.append('L2')
            if self.ANNnoneBox.isChecked():
                reg_types.append('None')
            reg_param = [i for i in self.annLambdaList.text().split(",")]

            dropout_probs = [float(c) for c in self.annDropoutList.text().split(",")]

            batchnorms = []
            if self.ANNbnyesBox.isChecked():
                batchnorms.append(True)
            if self.ANNbnnoBox.isChecked():
                batchnorms.append(False)

            solvers = []
            if self.ANNsvgBox.isChecked():
                solvers.append('svg')
            if self.ANNmomentumBox.isChecked():
                solvers.append('momentum')
            if self.ANNadamBox.isChecked():
                solvers.append('Adam')
            if self.ANNrmsBox.isChecked():
                solvers.append('RMSprop')

            max_epochs = [int(c) for c in self.annEpochsList.text().split(",")]

            lr = [float(c) for c in self.annLrList.text().split(",")]

            lr_decay = [float(i) for i in self.annLambdaList.text().split(",")]

            models['ANN'].param_grid = {
                'model__hidden_layers': hidden_layers, 'model__activation_function': activation_fcts,
                'model__loss_function': loss_fcts, 'model__reg_type': reg_types, 'model__reg_param': reg_param,
                'model__dropout_prob': dropout_probs, 'model__batchnorm': batchnorms, 'model__solver': solvers,
                'model__max_epochs': max_epochs, 'model__learning_rate': lr, 'model__decay': lr_decay,
                'model__val_split': [val_data]}

        if self.cnnBox.isChecked():
            models['CNN'] = ModelClass()

            embedding_sizes = [int(i) for i in self.cnnEList.text().split(",")]

            hidden_layers = []
            help = [experiment for experiment in self.cnnFList.text().split(";")]
            for experiment in help:
                experiment2 = []
                for layer in experiment.split(","):
                    layer = layer[1:-1] #remove brackets
                    experiment2.append([int(k) for k in layer.split(" ")])
                hidden_layers.append(experiment2)

            convolution_types = []
            if self.cnnConvSameBox.isChecked():
                convolution_types.append('same')
            if self.cnnConvValidBox.isChecked():
                convolution_types.append('valid')

            pooling = []
            if self.cnnPoolYesBox.isChecked():
                pooling.append('yes')
            if self.cnnPoolNoBox.isChecked():
                pooling.append('no')
            pooling_filter_sizes = [int(i) for i in self.cnnPoolFList.text().split(",")]
            pooling_filter_strides = [int(i) for i in self.cnnPoolSList.text().split(",")]

            activation_fcts = []
            if self.CNNreluBox.isChecked():
                activation_fcts.append('relu')
            if self.CNNsigmoidBox.isChecked():
                activation_fcts.append('sigmoid')
            if self.CNNtanhBox.isChecked():
                activation_fcts.append('tanh')

            loss_fcts = []
            if self.CNNhingeBox.isChecked():
                loss_fcts.append('hinge')
            if self.CNNentropyBox.isChecked():
                loss_fcts.append('binary_crossentropy')

            reg_types = []
            if self.CNNL1Box.isChecked():
                reg_types.append('L1')
            if self.CNNL2Box.isChecked():
                reg_types.append('L2')
            if self.CNNnoneBox.isChecked():
                reg_types.append('None')
            reg_param = [i for i in self.cnnLambdaList.text().split(",")]

            batchnorms = []
            if self.CNNbnyesBox.isChecked():
                batchnorms.append(True)
            if self.CNNbnnoBox.isChecked():
                batchnorms.append(False)

            solvers = []
            if self.CNNsvgBox.isChecked():
                solvers.append('svg')
            if self.CNNmomentumBox.isChecked():
                solvers.append('momentum')
            if self.CNNadamBox.isChecked():
                solvers.append('Adam')
            if self.CNNrmsBox.isChecked():
                solvers.append('RMSprop')

            max_epochs = [int(c) for c in self.cnnEpochsList.text().split(",")]

            lr = [float(c) for c in self.cnnLrList.text().split(",")]

            lr_decay = [float(i) for i in self.cnnLambdaList.text().split(",")]

            models['CNN'].param_grid = { 'hidden_layers': hidden_layers,
                'embedding_size': embedding_sizes,  'max_feat': [maxfeat],
                'convolution_type': convolution_types, 'pooling': pooling,
                'pooling_filter_size': pooling_filter_sizes,
                'pooling_filter_stride': pooling_filter_strides,
                'activation_function': activation_fcts,
                'loss_function': loss_fcts, 'reg_type': reg_types, 'reg_param': reg_param,
                'solver': solvers, 'batchnorm':batchnorms,
                'max_epochs': max_epochs, 'learning_rate': lr, 'decay': lr_decay,
                'val_split': [val_data]}

        if self.lstmBox.isChecked():
            models['LSTM'] = ModelClass()

            embedding_sizes = [int(i) for i in self.lstmEList.text().split(",")]

            hidden_layers = []
            help = [i.split(",") for i in self.lstmHList.text().split(";")]
            for string in help:
                hidden_layers.append([int(k) for k in string])

            dropout_probs = [float(c) for c in self.lstmDropoutList.text().split(",")]

            loss_fcts = []
            if self.lstmhingeBox.isChecked():
                loss_fcts.append('hinge')
            if self.lstmentropyBox.isChecked():
                loss_fcts.append('binary_crossentropy')

            reg_types = []
            if self.lstmL1Box.isChecked():
                reg_types.append('L1')
            if self.lstmL2Box.isChecked():
                reg_types.append('L2')
            if self.lstmnoneBox.isChecked():
                reg_types.append('None')
            reg_param = [i for i in self.lstmLambdaList.text().split(",")]

            solvers = []
            if self.lstmsvgBox.isChecked():
                solvers.append('svg')
            if self.lstmmomentumBox.isChecked():
                solvers.append('momentum')
            if self.lstmadamBox.isChecked():
                solvers.append('Adam')
            if self.lstmrmsBox.isChecked():
                solvers.append('RMSprop')

            max_epochs = [int(c) for c in self.lstmEpochsList.text().split(",")]

            lr = [float(c) for c in self.lstmLrList.text().split(",")]

            lr_decay = [float(i) for i in self.lstmLambdaList.text().split(",")]

            models['LSTM'].param_grid = { 'hidden_layers': hidden_layers,
                'embedding_size': embedding_sizes,  'max_feat': [maxfeat],
                'dropout_prob': dropout_probs,
                'loss_function': loss_fcts, 'reg_type': reg_types, 'reg_param': reg_param,
                'solver': solvers,
                'max_epochs': max_epochs, 'learning_rate': lr, 'decay': lr_decay,
                'val_split': [val_data]}

        # execute experiments
        if len(set(self.reviews.labels)) == 2:
            binary=True
        else:
            binary=False

        self.all_logs = []
        seed = self.seed
        K.clear_session()
        self.experiments = []

        X_train, X_test, idx_train, idx_test, y_train, y_test = train_test_split(
            self.reviews.cleantext[self.reviews.labelled_indices],
            self.reviews.labelled_data.index, self.reviews.labels, test_size=test_data,
            random_state=seed, stratify=self.reviews.labels)

        #save vars for ensemble models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        total = len(document_models) * len(ngram_ranges) * len(models.keys()) + 8
        i = 0
        for type in document_models:
            for n_grams in ngram_ranges:
                sample_type = 'stratified'
                vectoriser = modelling.vectorise(type, maxfeat, n_grams)

                for key in models.keys():
                    if key == 'CNN' or key == 'LSTM':
                        continue
                    print('\n\nBusy with ', type, n_grams, sample_type, key)
                    i += 1
                    if key == 'NB':
                        estimator = modelling.train_NB(type=type)
                        if type == 'tf-idf':
                            models[key].param_grid = {}  # if GaussianNB, no hyperparameters
                        else:
                            models[key].param_grid = nb_params  # reset if re-used afterwards
                    if key == 'SVM':
                        estimator = svm.SVC()
                    if key == 'LogReg':
                        estimator = LogisticRegression()
                    if key == 'ANN':
                        estimator = modelling.ANN()

                    pipeline = Pipeline(steps=[('vectorise', vectoriser),
                                                   ('to_dense', DenseTransformer()),
                                                   ('model', estimator)])

                    experiment = Experiment(algorithm=key, modelclass=ModelClass(),
                                                document_representation=type,
                                                ngram_range=n_grams, sampling_method=sample_type, maxfeat=maxfeat)

                    [experiment.modelclass.model, experiment.modelclass.crossval_score,
                     experiment.modelclass.best_params] = modelling.grid_search(pipeline, X_train, y_train,
                                                                                parameters=models[key].param_grid,
                                                                                metric=metric, nfolds=n_folds)
                    print('best params:', experiment.modelclass.best_params)

                    # evaluate on test set
                    try:
                        predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                                    'numerical': [
                                                        experiment.modelclass.model.predict_proba(X_test)]})
                    except:
                        predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                                    'numerical': None})
                    path = 'assets/confusion_'+str(i)+'.png'
                    micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                                    path=path,
                                                                                    return_type='all',
                                                                                    )
                    experiment.log_results(micro_auc, accuracy, precision, recall, f1)
                    print(experiment.modelclass.best_params, experiment.modelclass.crossval_score)
                    self.experiments.append(experiment)
                    self.evaluateprogressBar.setValue(i / total * 100)

        if self.cnnBox.isChecked():
            estimator = modelling.CNN()
            experiment = Experiment(algorithm='CNN', modelclass=ModelClass(),
                                            document_representation='Word embeddings',
                                            ngram_range='N/A', sampling_method='stratified', maxfeat=maxfeat)
            [experiment.modelclass.model, experiment.modelclass.crossval_score,
             experiment.modelclass.best_params] = modelling.grid_search(estimator, X_train, y_train,
                                                                        parameters=models['CNN'].param_grid,
                                                                        metric=metric, nfolds=n_folds)
            print('best params:', experiment.modelclass.best_params)

            # evaluate on test set
            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                            'numerical': None})
            i = i+1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            print(experiment.modelclass.best_params, experiment.modelclass.crossval_score)
            self.experiments.append(experiment)
            self.evaluateprogressBar.setValue(i / total * 100)

        if self.lstmBox.isChecked():
            estimator = modelling.LSTM()
            experiment = Experiment(algorithm='LSTM', modelclass=ModelClass(),
                                    document_representation='Word embeddings',
                                    ngram_range='N/A', sampling_method='stratified',
                                    maxfeat=maxfeat)
            [experiment.modelclass.model, experiment.modelclass.crossval_score,
             experiment.modelclass.best_params] = modelling.grid_search(estimator, X_train, y_train,
                                                                        parameters=models['LSTM'].param_grid,
                                                                        metric=metric, nfolds=n_folds)
            print('best params:', experiment.modelclass.best_params)

            # evaluate on test set
            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test)],
                                            'numerical': None})
            i = i + 1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            print(experiment.modelclass.best_params, experiment.modelclass.crossval_score)
            self.experiments.append(experiment)
            self.evaluateprogressBar.setValue(i / total * 100)

        # test lexicon-based methods
        if self.sentiwordnetBox.isChecked():
            experiment = Experiment(algorithm='Sentiwordnet', modelclass=ModelClass(), document_representation='n/a',
                                    ngram_range='n/a', sampling_method='n/a', maxfeat='n/a')
            swn = modelling.Sentiwordnet()
            experiment.modelclass.model = swn.fit(X_train, y_train)
            experiment.modelclass.crossval_score = "n/a"
            experiment.modelclass.best_params = "n/a"

            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test, binary=binary)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': None})

            i = i + 1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            self.experiments.append(experiment)
        self.evaluateprogressBar.setValue(i / total * 100)

        if self.patternBox.isChecked():
            experiment = Experiment(algorithm='Pattern', modelclass=ModelClass(),
                                    document_representation='n/a',
                                    ngram_range='n/a', sampling_method='n/a', maxfeat='n/a')
            ptn = modelling.Pattern_sentiment()
            experiment.modelclass.model = ptn.fit(X_train, y_train)
            experiment.modelclass.crossval_score = "n/a"
            experiment.modelclass.best_params = "n/a"

            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test, binary=binary)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': None})
            i = i + 1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            self.experiments.append(experiment)
        self.evaluateprogressBar.setValue(i / total * 100)

        if self.huliuBox.isChecked():
            experiment = Experiment(algorithm='Hu and Liu', modelclass=ModelClass(),
                                    document_representation='n/a',
                                    ngram_range='n/a', sampling_method='n/a', maxfeat='n/a')
            hl = modelling.Hu_liu_sentiment()
            experiment.modelclass.model = hl.fit(X_train, y_train)
            experiment.modelclass.crossval_score = "n/a"
            experiment.modelclass.best_params = "n/a"

            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test, binary=binary)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': None})
            i = i + 1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            self.experiments.append(experiment)
        self.evaluateprogressBar.setValue(i / total * 100)

        if self.vaderBox.isChecked():
            experiment = Experiment(algorithm='Vader', modelclass=ModelClass(), document_representation='n/a',
                                    ngram_range='n/a', sampling_method='n/a', maxfeat='n/a')
            vdr = modelling.Vader()
            experiment.modelclass.model = vdr.fit(X_train, y_train)
            experiment.modelclass.crossval_score = "n/a"
            experiment.modelclass.best_params = "n/a"
            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(X_test, binary=binary)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(X_test, binary=binary)],
                                            'numerical': None})
            print(predictions)
            i = i + 1
            path = 'assets/confusion_' + str(i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(y_test, predictions,
                                                                                        path=path,
                                                                                        return_type='all',
                                                                                        )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            self.experiments.append(experiment)

        log = pd.DataFrame([d.__dict__ for d in self.experiments], index=range(len(self.experiments)))
        log['best_params'] = [d.modelclass.best_params if d.algorithm not in ['Sentiwordnet', 'Vader', 'Hu and Liu', 'Pattern']
         else "N/A" for d in self.experiments]
        name = datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+str(seed)
        log.to_csv("log/" + name + ".csv")
        self.current_i = i
        self.current_binary = binary

        self.all_logs.append(log)

        self.evaluateprogressBar.setValue(100)

        self.stack.setCurrentIndex(5)
        self.comparetable.clearContents()
        self.comparetable.setHorizontalHeaderItem(0, QTableWidgetItem("Experiment"))
        self.comparetable.setHorizontalHeaderItem(1, QTableWidgetItem("Model"))
        self.comparetable.setHorizontalHeaderItem(2, QTableWidgetItem("Doc. Representation"))
        self.comparetable.setHorizontalHeaderItem(3, QTableWidgetItem("n-gram Range"))
        self.comparetable.setHorizontalHeaderItem(4, QTableWidgetItem("Sampling method"))
        self.comparetable.setHorizontalHeaderItem(5, QTableWidgetItem("AUC"))
        self.comparetable.setHorizontalHeaderItem(6, QTableWidgetItem("Accuracy"))
        self.comparetable.setHorizontalHeaderItem(7, QTableWidgetItem("F1-Score"))
        self.comparetable.setHorizontalHeaderItem(8, QTableWidgetItem("Precision"))
        self.comparetable.setHorizontalHeaderItem(9, QTableWidgetItem("Recall"))
        self.comparetable.setRowCount(len(self.experiments))
        for row in range(len(self.experiments)):
            self.comparetable.setItem(row, 0, QTableWidgetItem(str(row)))
            self.comparetable.setItem(row, 1, QTableWidgetItem(self.experiments[row].algorithm))
            self.comparetable.setItem(row, 2, QTableWidgetItem(self.experiments[row].document_representation))
            self.comparetable.setItem(row, 3, QTableWidgetItem(str(self.experiments[row].ngram_range)))
            self.comparetable.setItem(row, 4, QTableWidgetItem(str(self.experiments[row].sampling_method)))
            self.comparetable.setItem(row, 5, QTableWidgetItem("{:1.4f}".format(self.experiments[row].auc)))
            self.comparetable.setItem(row, 6, QTableWidgetItem("{:1.4f}".format(self.experiments[row].accuracy)))
            self.comparetable.setItem(row, 7, QTableWidgetItem("{:1.4f}".format(self.experiments[row].f1)))
            self.comparetable.setItem(row, 8, QTableWidgetItem("{:1.4f}".format(self.experiments[row].precision)))
            self.comparetable.setItem(row, 9, QTableWidgetItem("{:1.4f}".format(self.experiments[row].recall)))

            try:
                self.experiments[row].__dict__.update(self.experiments[row].modelclass.__dict__)
            except:
                pass
    ################################################
    # Evaluate page functions
    ################################################
    @Slot()
    def on_backtomodelButton_clicked(self):
        self.seed = random.randint(0, 200)  # randomise seed for next set of experiments
        print("new seed", self.seed)
        self.evaluateprogressBar.setValue(0)
        self.stack.setCurrentIndex(4)

    @Slot()
    def on_comparetable_itemSelectionChanged(self):
        if len(self.comparetable.selectionModel().selectedRows()) == 1:
            self.ensembleButton.setEnabled(False)
            experiment_nr = int(self.comparetable.item(self.comparetable.currentRow(), 0).text())
            try:
                self.sel_experiment = self.experiments[experiment_nr]

                if self.sel_experiment.modelclass == 'n/a':
                    label = "Algorithm:\t{0}\n(Lexicon-based method)".format(self.sel_experiment.algorithm)
                else:
                    label = "Algorithm:\t{0}\nHyperparameters:\t{1}\nCrossvalidated score:\t{2}".format(
                        self.sel_experiment.algorithm,
                        self.sel_experiment.modelclass.best_params,
                        self.sel_experiment.modelclass.crossval_score)
                self.describeLabel.setText(label)
                CM_path = 'assets/confusion_'+str(experiment_nr+1)+'.png'
                CM_image_profile = QtGui.QImage(CM_path)
                self.confusion_matrix.setPixmap(QtGui.QPixmap.fromImage(CM_image_profile))
                self.selectModelButton.setEnabled(True)
            except:
                pass

        elif len(self.comparetable.selectionModel().selectedRows()) > 1:
            self.ensembleButton.setEnabled(True)

    @Slot()
    def on_ensembleButton_clicked(self):
        self.ew.exec()
        if self.ew.result == 1:
            #if not cancelled, generate ensemble
            base_learners_index = [int(self.comparetable.item(row.row(), 0).text())
                             for row in  self.comparetable.selectionModel().selectedRows()]
            base_learners = []
            accuracies = []
            for i in base_learners_index:
                name = str(self.experiments[i].algorithm)+str(self.experiments[i].document_representation)+str(self.experiments[i].ngram_range)
                base_learners.append(
                    (name,
                     self.experiments[i].modelclass.model)
                )
                if self.ew.configuration['combinationMethod'] == 'weighted':
                    if self.experiments[i].modelclass.crossval_score != 'n/a':
                        accuracies.append(self.experiments[i].modelclass.crossval_score)
                    else:
                        #if crossval accuracy not available (lexicon-based models), calculate "training accuracy"
                        predictions = pd.DataFrame(
                            {'labels': [self.experiments[i].modelclass.model._predict(self.X_train, binary=self.current_binary)],
                            'numerical': None})
                        micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(self.y_train, predictions,
                                                                                        path=None,
                                                                                        return_type='all',
                                                                                        )
                        accuracies.append(accuracy)
            ensemble_model = modelling.generate_ensemble(base_learners, self.ew.configuration, accuracies)
            experiment = Experiment(algorithm='Ensemble '+str(base_learners_index), modelclass=ModelClass(),
                                    document_representation='n/a',
                                    ngram_range='n/a', sampling_method='stratified', maxfeat='n/a')
            experiment.modelclass.model = ensemble_model.fit(self.X_train, np.array(self.y_train))
            experiment.modelclass.crossval_score = 'n/a'
            params = self.ew.configuration.copy()
            params['models'] = [i[0] for i in base_learners]
            experiment.modelclass.best_params = params.copy()

            # evaluate on test set
            try:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(self.X_test)],
                                            'numerical': [
                                                experiment.modelclass.model.predict_proba(self.X_test)]})
            except:
                predictions = pd.DataFrame({'labels': [experiment.modelclass.model.predict(self.X_test)],
                                            'numerical': None})
            self.current_i = self.current_i + 1
            path = 'assets/confusion_' + str(self.current_i) + '.png'
            micro_auc, accuracy, precision, recall, f1 = modelling.evaluate(self.y_test, predictions,
                                                                            path=path,
                                                                            return_type='all',
                                                                            )
            experiment.log_results(micro_auc, accuracy, precision, recall, f1)
            print(experiment.modelclass.best_params, experiment.modelclass.crossval_score)
            self.experiments.append(experiment)

            log = pd.DataFrame([d.__dict__ for d in self.experiments], index=range(len(self.experiments)))
            log['best_params'] = [
                d.modelclass.best_params if d.algorithm not in ['Sentiwordnet', 'Vader', 'Hu and Liu', 'Pattern']
                else "N/A" for d in self.experiments]
            name = datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + str(self.seed)
            log.to_csv("log/" + name + ".csv")

            self.all_logs.append(log)
            self.all_logs_df = pd.concat(self.all_logs, axis=0)
            self.all_logs_df.to_csv("log/" + datetime.now().strftime("%Y%m%d-%H%M%S") + 'all_folds.csv')

            self.evaluateprogressBar.setValue(100)

            self.stack.setCurrentIndex(5)
            self.comparetable.clearContents()
            self.comparetable.setHorizontalHeaderItem(0, QTableWidgetItem("Experiment"))
            self.comparetable.setHorizontalHeaderItem(1, QTableWidgetItem("Model"))
            self.comparetable.setHorizontalHeaderItem(2, QTableWidgetItem("Doc. Representation"))
            self.comparetable.setHorizontalHeaderItem(3, QTableWidgetItem("n-gram Range"))
            self.comparetable.setHorizontalHeaderItem(4, QTableWidgetItem("Sampling method"))
            self.comparetable.setHorizontalHeaderItem(5, QTableWidgetItem("AUC"))
            self.comparetable.setHorizontalHeaderItem(6, QTableWidgetItem("Accuracy"))
            self.comparetable.setHorizontalHeaderItem(7, QTableWidgetItem("F1-Score"))
            self.comparetable.setHorizontalHeaderItem(8, QTableWidgetItem("Precision"))
            self.comparetable.setHorizontalHeaderItem(9, QTableWidgetItem("Recall"))
            self.comparetable.setRowCount(len(self.experiments))
            for row in range(len(self.experiments)):
                self.comparetable.setItem(row, 0, QTableWidgetItem(str(row)))
                self.comparetable.setItem(row, 1, QTableWidgetItem(self.experiments[row].algorithm))
                self.comparetable.setItem(row, 2, QTableWidgetItem(self.experiments[row].document_representation))
                self.comparetable.setItem(row, 3, QTableWidgetItem(str(self.experiments[row].ngram_range)))
                self.comparetable.setItem(row, 4, QTableWidgetItem(str(self.experiments[row].sampling_method)))
                self.comparetable.setItem(row, 5, QTableWidgetItem("{:1.4f}".format(self.experiments[row].auc)))
                self.comparetable.setItem(row, 6, QTableWidgetItem("{:1.4f}".format(self.experiments[row].accuracy)))
                self.comparetable.setItem(row, 7, QTableWidgetItem("{:1.4f}".format(self.experiments[row].f1)))
                self.comparetable.setItem(row, 8, QTableWidgetItem("{:1.4f}".format(self.experiments[row].precision)))
                self.comparetable.setItem(row, 9, QTableWidgetItem("{:1.4f}".format(self.experiments[row].recall)))

                try:
                    self.experiments[row].__dict__.update(self.experiments[row].modelclass.__dict__)
                except:
                    pass

    @Slot()
    def on_selectModelButton_clicked(self):
        # Save selected model
        filename = './saved_models/' + self.sel_experiment.algorithm + str(datetime.now())[:10] + '.joblib'
        if self.sel_experiment.modelclass == 'n/a':
            pass  # don't save lexicon-based methods (no parameters)
        else:
            dump(self.sel_experiment.modelclass.model, filename)

        if self.viewResultsBox.isChecked():
            # launch Dash App to display summaries
            print("Deploying model...")
            self.finalpredictions = self.sel_experiment.modelclass.model.predict(self.reviews.cleantext)

            try:
                if np.array(self.finalpredictions)[0].shape[0] == 1:  # if funny 2D output of NN model
                    self.finalpredictions = [pred[0] for pred in self.finalpredictions]
            except:
                pass

            # Launch Dash App
            print("Linking supplementary data...")
            try:
                final = pd.DataFrame({'Client_num': self.reviews.data[self.clients.links['reviews_side']],
                                      'Review': self.reviews.text,
                                      'Sentiment': self.finalpredictions},
                                     )
                final = pd.concat([final, self.reviews.df], axis=1)
                my_dash = Dashboard(final_predictions=final, reviews=self.reviews, customer_data=self.clients)
            except:
                final = pd.DataFrame({'Client_num': None,
                                      'Review': self.reviews.text,
                                      'Sentiment': self.finalpredictions},
                                     )
                final = pd.concat([final, self.reviews.df], axis=1)
                my_dash = Dashboard(final_predictions=final, reviews=self.reviews, customer_data=None)

            print("Closing window...")
            self.close()
            # self.quit()
            print("Launching Dash App...")
            my_dash.deploy()

    def closeEvent(self, event):
        self.deleteLater()

'''Deploy App'''
if __name__ == '__main__':
    app = QApplication(sys.argv)

    qtmodern.styles.dark(app)
    mw = qtmodern.windows.ModernWindow(MainWindow())
    mw.show()

    sys.exit(app.exec_())
