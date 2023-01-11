# Channel Selection for motor imagery BCI

## short description
This repository contains code about the selection of optimal channels in a BCI based on motor imagery.
The main EEG data exploited for these studies are already available online and well-known in scientific literature.

The scripts provided here were implemented in Matlab R2020b, but they could be used with previous versions of Matlab as well.
They include the "filter bank common spatial pattern" approach and classifiers like SVM and Bayesian ones.

## less short description
In building a wearable brain-computer interface based on motor imagery (MI), channel selection/reduction is a crucial topic.
Previous studies suggested that even a single channel could be used, but that stategey was not tested.
Therefore, although first evidence seemed to confirm that, further experiments have denied such a possibility and today it appears that at least 3 channels are needed.

The code provided in this repository thus applies to the classification of EEG signals associated with motor imagery in these conditions: 
- attempt to select a single optimal channel
- investigation of the relation between number of channels and classification accuracy

The datasets exploited in these studies are:
- BCI competition IV dataset 2a: https://www.bbci.de/competition/iv/#datasets
- BCI competition III dataset 3a: https://www.bbci.de/competition/iii/#data_set_iiia

A folder with the scripts of FBCSP algorithm is also included. These are needed for the investigation of the relation between number of channels and classification accuracy.

## how to use
In order to reproduce our results as a starting point for your analyses, you can simply follow these steps:
1. download a folder from this repository depending on the study of interest
    it would be recommended to keep its structure as is, and include to path eventually subfolders
    for channel selection on datasets 2a and 3a you must also download the "FBCSP algorithm" folder
2. download data from above links and put it in the downloaded folder
    you could include it as a subfolder and add it to path
3. look for the main script and play it
4. eventually install missing tools (though you should have all functions already)
5. enjoy

## further details

1. Arpaia, P., Donnarumma, F., Esposito, A. and Parvis, M., 2021. Channel selection for optimal EEG measurement in motor imagery-based brain-computer interfaces. International Journal of Neural Systems, 31(03), p.2150003, [doi: 10.1142/S0129065721500039](https://doi.org/10.1142/S0129065721500039)
2. Angrisani, L., Arpaia, P., Esposito, A., Gargiulo, L., Natalizio, A., Mastrati, G., Moccaldi, N. and Parvis, M., 2021. Passive and active brain-computer interfaces for rehabilitation in health 4.0. Measurement: Sensors, 18, p.100246, [doi: 10.1016/j.measen.2021.100246](https://doi.org/10.1016/j.measen.2021.100246)
3. Angrisani, L., Arpaia, P., Donnarumma, F., Esposito, A., Moccaldi, N. and Parvis, M., 2019, May. Metrological performance of a single-channel Brain-Computer Interface based on Motor Imagery. In 2019 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) (pp. 1-5). IEEE, [doi: 10.1109/I2MTC.2019.8827168](https://doi.org/10.1109/I2MTC.2019.8827168)
4. "Motor imagery brain-computer interface and extended reality" [project on ResearchGate](https://www.researchgate.net/project/Motor-imagery-brain-computer-interface-and-extended-reality)

