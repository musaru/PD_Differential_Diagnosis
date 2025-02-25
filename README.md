# PD_Differential_Diagnosis

##Preprocessing
The raw gyroscopic data were preprocessed to ensure consistency and readiness for analysis. The data are given in .mat file (Matlab format) and contain the following information: Symptom ('CTRL', 'PD', 'MSA', or 'PSP'), '$gyroThumb\_\{X,Y,Z\}$': x, y, z axes of the gyroscope in the thumb, '$gyroIndex\_\{X,Y,Z\}$': x, y, z axes of the gyroscope in the index finger, 'personID': person code, 'trialID': trial code, Sampling rate (Hz) (but always 200 Hz). In this study, this .mat file was converted to a csv file and the triaxial gyroscope data $(gyroThumb\_\{X, Y, Z\}$, $gyroIndex\_\{X,Y, Z\})$ was used in the analysis. Example visualizations of the dataset signals for each disease group are shown in Figure \ref{Figure:2}
