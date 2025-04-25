# heart_sounds
Detect abnormal heart sounds

Original Abstract — The objective of this project was to classify heart
sounds as normal or abnormal from raw audio files using various
machine learning techniques. The files were initially processed to
remove unwanted artifacts and retain useful information.
Traditional feature extraction of relative peaks and time duration
between relative peaks was attempted. Dynamic Time Warping
was also attempted as a method of feature extraction. Mel-
frequency cepstral coefficients were the most effective features
used. Classification techniques used were k-Nearest Neighbors
and Support Vector Machines. Ensemble methods were then
added to increase accuracy. The final test accuracy obtained with
the Mel-frequency cepstral coefficients features and kNN was
49%. Due to computer memory and capacity, not all of the
training data was used. In the future, using all the training data
with a better performing computer and increasing the precision
of the abnormal instances, will increase the accuracy.

Updates -- Testing CNN and LSTM Classification.
