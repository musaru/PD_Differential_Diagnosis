# PD_Differential_Diagnosis
'''Dataset'''
We used publicly available data from the Finger Tapping (FT) task, collected in Belic's study [1] and deposited on GitHub. A total of 267 samples, recorded via a gyroscope, were made available. 
[1] M. Beli´c, Z. Radivojevi´c, V. Bobi´c, V. Kosti´c, and M. Ðuri´c-Joviˇci´c, “Quick computer aided differential diagnostics based on repetitive fin-
ger tapping in parkinson’s disease and atypical parkinsonisms,” Heliyon, vol. 9, no. 4, 2023

'''Preprocessing'''
The raw gyroscopic data were preprocessed to ensure consistency and readiness for analysis. The data are given in .mat file (Matlab format) and contain the following information: Symptom ('CTRL', 'PD', 'MSA', or 'PSP'), '$gyroThumb\_\{X,Y,Z\}$': x, y, z axes of the gyroscope in the thumb, '$gyroIndex\_\{X,Y,Z\}$': x, y, z axes of the gyroscope in the index finger, 'personID': person code, 'trialID': trial code, Sampling rate (Hz) (but always 200 Hz). In this study, this .mat file was converted to a csv file and the triaxial gyroscope data $(gyroThumb\_\{X, Y, Z\}$, $gyroIndex\_\{X,Y, Z\})$ was used in the analysis. Example visualizations of the dataset signals for each disease group are shown in Figure \ref{Figure:2}
