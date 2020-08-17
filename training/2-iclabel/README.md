## README file for the use of iclabel preprocessing for the whole Neureka database

The main script that must be run is "run_icalabel.m"  (number of start and end subject must be defined as an input in order to be able to parallelize it)
The architecture of the files must be the same as in the initial dataset 

The level for artifact removal has been set to correlation 0.6 (everything above this are rejected as artifact).

In order to run the code you need eeglab2019_1, with plugins Biosig, Cleanrawdata and IClabel v 1.2.4 preinstalled
You need to replace the following functions of eeglab: 
  - pop_runica.m   (line 385 adapted in order to have full rank)
  - pop_clean_rawdata.m  (line 125 adapted in order to be able to load the same preselected settings for Cleanrawdata plugin).

The file input.mat includes the preselected settings for Cleanrawdata plugin

The file neureca.locs includes the positions and different namings of the electrodes in Neureka challenge

The files findRecording.m and getContent.m are used in order to browse among different subjects
