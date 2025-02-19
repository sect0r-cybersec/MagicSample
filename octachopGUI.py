#!/usr/bin/env python3.10
## Shebang for linux/unix systems

"""
Note:

The current version of this script only works with Python 3.10.

This is to do with a dependency issue within spleeter. Until that is fixed, please run
the script with Python 3.10
"""

## Ignores verbose TensorFlow warnings
import warnings
warnings.filterwarnings("ignore")

## For testing purposes
import time

## Needed to work with directories
import os

## Needed for PyQT
import sys

## I chose PyQT as my window manager option for GUI
##from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QLineEdit, QProgressBar, QPushButton, QCheckBox, QSlider, QComboBox, QTreeWidget, QTreeView
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtQuick import *

## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

## Required for manipulating audio files, beat detection
import librosa.core, librosa.beat, librosa.effects

## Required to export numpy floating point array objects as sound (.wav files)
import soundfile as sf

## librosa returns data from methods as numpy array objects, so numpy is needed to manipulate that data
import numpy as np

## Spleeter is Deezer's stem separation ML model.
## I'm using a pretrained algorithm supplied with the library to separate stems.
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

input_path = ("test_data/omen.wav")
stems_path =("temp")

"""
5 stems are:
bass
drums
other
piano
vocals
"""

class Window(QWidget):

    window = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def disableInputs(self, boolean):
        self.stems_checkbox.setEnabled(not boolean)
        self.bpm_checkbox.setEnabled(not boolean)
        self.pitch_checkbox.setEnabled(not boolean)

        self.input_filepath.setReadOnly(boolean)
        self.output_filepath.setReadOnly(boolean)
        self.output_foldername.setReadOnly(boolean)

        self.sensitivity_slider.setEnabled(not boolean)
        self.output_format.setEnabled(not boolean)
        self.start_button.setEnabled(not boolean)

    def set_lineinp_filepath_old(self, tree, text_input):
        index = tree.selectedIndexes()[0]
        path = tree.model().fileInfo(index)
        absolute_path = (path.absoluteFilePath())
        text_input.setText(absolute_path)

    def set_lineinp_filepath(self, tree, text_input):
        indexes = tree.selectedIndexes()
        files = []
        for index in indexes:
            path = tree.model().fileInfo(index)
            absolute_path = (path.absoluteFilePath())
            files.append(absolute_path)
        print(files)
        text_input.setText(files)

    def run_backend(self):

        ## Disable inputs

        self.disableInputs(True)

        ## Obtain all current values from inputs

        input_filepath = self.input_filepath.text()
        output_filepath = self.output_filepath.text()
        output_foldername = self.output_foldername.text()
        extension = self.output_format.currentText()
        stems_checkbox = self.stems_checkbox.isChecked()
        bpm_checkbox = self.bpm_checkbox.isChecked()
        pitch_checkbox = self.pitch_checkbox.isChecked()

        self.thread = QThread()
        self.worker = Worker(input_filepath, output_filepath, output_foldername, extension, stems_checkbox, bpm_checkbox)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        ## Enable inputs again

        self.thread.finished.connect(lambda: self.disableInputs(False))

    def setup_ui(self):
        self.setWindowTitle("OctaChop")
        self.setWindowIcon(QIcon("icons/octopus.png"))

        self.valid_ext = ("wav", "flac", "ogg")
    
        layout = QGridLayout()
        layout.fillWidth = True
        layout.fillHeight = True

        self.input_filepath = QLineEdit()
        self.input_filepath.setPlaceholderText("Input filepath...")
        layout.addWidget(self.input_filepath, 6, 0, 1, 6)

        self.output_filepath = QLineEdit()
        self.output_filepath.setPlaceholderText("Output path...")
        layout.addWidget(self.output_filepath, 6, 6, 1, 6)

        self.input_model = QFileSystemModel()
        self.input_model.setRootPath("")

        self.input_tree = QTreeView()
        self.input_tree.setModel(self.input_model)
        self.input_tree.setAnimated(True)
        ## Set to extended-selection
        self.input_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.input_tree.clicked.connect(lambda: self.set_lineinp_filepath(self.input_tree, self.input_filepath))
        layout.addWidget(self.input_tree, 0, 0, 6, 6)

        self.output_model = QFileSystemModel()
        self.output_model.setRootPath("")

        self.output_tree = QTreeView()
        self.output_tree.setModel(self.output_model)
        self.output_tree.setAnimated(True)
        self.output_tree.clicked.connect(lambda: self.set_lineinp_filepath_old(self.output_tree, self.output_filepath))
        layout.addWidget(self.output_tree, 0, 6, 6, 6)

        self.stems_checkbox = QCheckBox("Split to stems?")
        layout.addWidget(self.stems_checkbox, 7, 5)

        self.bpm_checkbox = QCheckBox("Detect bpm?")
        layout.addWidget(self.bpm_checkbox, 7, 6)

        self.pitch_checkbox = QCheckBox("Detect pitch?")
        layout.addWidget(self.pitch_checkbox, 7, 7)

        self.sensitivity_slider_label = QLabel("Sensitivity")
        layout.addWidget(self.sensitivity_slider_label, 7, 0)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.sensitivity_slider, 7, 1, 1, 4)

        self.output_foldername = QLineEdit()
        ## Adds greyed out text to show the user to input an output filename
        self.output_foldername.setPlaceholderText("Output foldername...")
        layout.addWidget(self.output_foldername, 7, 8, 1, 2)

        self.output_format = QComboBox()
        ##valid_ext = ("wav", "flac", "ogg")
        for ext in self.valid_ext:
            self.output_format.addItem(ext)
        layout.addWidget(self.output_format, 7, 10)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar, 8, 0, 1, 12)

        self.start_button = QPushButton("Start")

        ## Links access to all other elements of the GUI
        self.start_button.clicked.connect(lambda: self.run_backend())

        layout.addWidget(self.start_button, 7, 11)


        ## This portion of code enables the window to stretch along the x axis,
        ## as well as the file explorer portion of the y axis
        filetree_columns = 11
        filetree_rows = 5
        for i in range(filetree_columns):
            layout.setColumnStretch(i, 1)
        for i in range(filetree_rows):
            layout.setRowStretch(i, 1)

        total_columns = 11
        total_rows = 8
        min_width = 80
        min_height = 40
        for i in range(total_columns):
            layout.setColumnMinimumWidth(i, min_width)
        for i in range(total_rows):
            layout.setRowMinimumHeight(i, min_height)

        self.setLayout(layout)

class Worker(QObject):

    def __init__(self, input_filepath, output_filepath, output_foldername, extension, stems_checkbox, bpm_checkbox):
        super(Worker, self).__init__()
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.output_foldername = output_foldername
        self.extension = extension
        self.stems_checkbox = stems_checkbox
        self.bpm_checkbox = bpm_checkbox

    finished = pyqtSignal()
    sub_progress = pyqtSignal(int)
    main_progress = pyqtSignal(int)
    
    def run(self):

        ## Methods for the class
        def detect_bpm(audio_data, sample_rate):
            bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate, sparse=False)
            bpm = (np.round(bpm_numpy_array).astype(int))[0]
            return bpm

        ##def detect_pitch(audio_data, sample_rate):


        def invert_numpy_array(waveform_dict):
            invert_dict = {}
            dict_keys = waveform_dict.keys()
            for key in dict_keys:
                waveform = waveform_dict.get(key)
                swapped_axis = np.swapaxes(waveform, 0, 1)
                invert_dict[key] = swapped_axis
            return invert_dict

        def split_to_stems(waveform, sample_rate):
            
            ## Instantiates 5 stem spleeter separator object
            ## Multiprocessing is disabled, otherwise windows has problems
            separator = Separator("spleeter:5stems-16kHz", multiprocess=False)
            
            waveform = np.swapaxes(waveform, 0, 1)
            
            ## Separator returns a dictionary object with the stem name as key (e.g. drums, bass, vocals), and the waveform as the value
            stems_dictionary = separator.separate(waveform)

            stems_dictionary = invert_numpy_array(stems_dictionary)

            ## Return stem waveform dictionary with axes swapped so it can be manipulated by librosa
            return stems_dictionary

        def write_waveform_to_file(waveform, sample_rate, path, name, extension):
            ## Soundfile and librosa use opposite axes for channels x data
            soundfile_waveform = np.swapaxes(waveform, 0, 1)
            PCM = "PCM_24"
            ## Concatenate filename and extension
            out_path = ("{0}/{1}.{2}".format(path, name, extension))
            sf.write(out_path, soundfile_waveform, sample_rate, PCM)

        def ext_isValid(file):
            for ext in self.valid_ext:
                file_ext = ((file.split("."))[-1]).lower()
                if file_ext == ext: ## If file extension is equal to one as specified in tuple...
                    return True
                else:
                    return False

        def sample(waveform, sample_rate, filename, sens):
            sample_index = librosa.effects.split(waveform, top_db=sens)
            for sample in sample_index:
                sample_start = sample[0]
                sample_end = sample[1]
                sample_waveform = waveform[:, sample_start:sample_end]
                write_waveform_to_file(sample_waveform, sample_rate, filename)

        input_files = []

        """
        if os.path.isdir(self.input_filepath): ## If user selected folder (multiple files)...
            files = os.listdir(self.input_filepath) ## Lists files in current directory. Is not recursive
            for file in files: ## For each file...
                good_file = os.path.abspath(file)
                input_files.append(good_file)
                    
        """
        ##elif os.path.isfile(self.input_filepath):

        print(input_filepath)

        return

        good_file = os.path.abspath(self.input_filepath)
        input_files.append(good_file)
            
        absolute_output_folder = os.path.join(self.output_filepath, self.output_foldername)

        if os.path.isdir(absolute_output_folder) == False:
            os.mkdir(absolute_output_folder)
        
        for file in input_files:
            
            waveform, sample_rate = librosa.load(file, sr=None, mono=False)

            if self.bpm_checkbox == True: ## If user wants bpm detection...
                bpm = detect_bpm(waveform, sample_rate)
            else:
                bpm = None
            
            if self.stems_checkbox == True: ## If user wants to split track to stems...
                stems = split_to_stems(waveform, sample_rate)
                stem_names = stems.keys()

            elif self.stems_checkbox == False:
                stem_names = ["Sample"]

            for key in stem_names:

                if self.stems_checkbox == True: ## If user wants to split track to stems...

                    stem_waveform = stems.get(key)
                    stem_path = os.path.join(absolute_output_folder, key)
                    if os.path.isdir(stem_path) == False:
                        os.mkdir(stem_path)

                filename = key
                    
                sample_index = librosa.effects.split(stem_waveform, top_db=10)
                count = 1
                for sample in sample_index:
                    sample_start = sample[0]
                    sample_end = sample[1]
                    sample_waveform = stem_waveform[:, sample_start:sample_end]

                    if self.bpm_checkbox == True:
                        filename = filename + " " + bpm + "bpm"

                    filename = filename + " " + count + "." + extension
                    write_waveform_to_file(sample_waveform, sample_rate, stem_path, filename, self.extension)
                    count += 1

        self.finished.emit()

def main():

    ## Prints errors to command line, comment out when not in use
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook


    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
