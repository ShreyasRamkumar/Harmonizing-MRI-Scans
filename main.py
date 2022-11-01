from dataset_prep import Preprocessing

preprocessing = Preprocessing()

preprocessing.createMRIMask("training\scanner2_sub-CC110069_T1w.nii")
preprocessing.correctBias("training\scanner2_sub-CC110069_T1w.nii")
