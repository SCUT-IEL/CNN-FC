# EEG-based Auditory Attention Detection via Frequency and Channel Neural Attention
This repository contains the python scripts developed as a part of the work presented in the paper "EEG-based Auditory Attention Detection via Frequency and Channel Neural Attention"

## Getting Started
These files are extracted from the project stream, which contain the complete model and some training parameters. In order to run the project, users need to write training and testing code, and use private or public datasets.

The public [KUL dataset](https://zenodo.org/record/3997352#.X_K7HmQzZ6J) is used in the paper. The dataset itself comes with matlab processing program, please adjust it according to your own needs.

## Model
![image](https://github.com/Enze-github/SCUT_-CNN-FC/blob/main/CNN-FC.png)
A schematic diagram of the proposed CNN classifier of five components with frequency and channel neural attention (CNN-FC): (A) input multi-channel EEG signals, and speech envelopes as the auditory stimulus references; (B) filter bank for EEG signals; (C1) 3D feature extraction of multi-channel EEG signals, (C2), frequency attention module, and (C3) channel attention module; (D), envelope extraction for speech streams, and (E), a convolutional neural network. The CNN-FC model is trained to detect the attended speaker, either speaker I or II, from the EEG signals. Note: speech streams of speaker I and II are denoted in red and green, while the EEG signals of the listener are denoted in blue.

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Contact
Enze Su, Shien-Ming Wu School of Intelligent Engineering, South China University of Technology, Guangzhou, Guangdong Province, China.

E-mail: enzesu@hotmail.com
