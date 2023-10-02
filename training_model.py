import cv2
import time
import os
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from fastai.learner import load_learner











# Redefine the function that automatically links images and labels
def label_func(fn): return "C:/labels/f{fn.stem}_P{fn.suffix}"
# Load.pkl file with the trained model.
learn = load_learner('/content/gdrive/MyDrive/TrainingDataset/MyOwnModel.pkl')

# select the MHD image that you want to segment
testFile = '/content/gdrive/MyDrive/Test/TestImage1.mhd'
if testFile.endswith('.mhd'):
    # If it is a MHD file, go on
    MRI = sitk.ReadImage(testFile)
# Reorient image to ensure the code works properly. Be
#consistent with the orientation used in Appendix 2.
    MRI = sitk.DICOMOrient(MRI,"LPI")
    prediction = np.empty(sitk.GetArrayFromImage(MRI).shape)
    for i in range(sitk.GetArrayFromImage(MRI).shape[2]):
        # for each one of the 2D sagittal slices of the MRI volume
        # create a RGB image from the gray level image
        slice2D = sitk.GetArrayFromImage(MRI[i,:,:])
        stacked_img = (np.dstack((slice2D,slice2D,slice2D))).astype(np.uint8)
        # Call the learner and do the prediction(this is the core of the
        #method)
        pred = learn.predict(stacked_img)
        prediction[:,:,i] = np.array(pred[0])
# Create your own folder for the prediction and change the directory
#to yours
save_dir = '/content/gdrive/MyDrive/Predictions/'
# This would be the name of your segmentation file
name = "TestImage1_pred.mha"
#Copy the header information from the MRI image and create 3D
#image of the segmentation
image3D = sitk.GetImageFromArray(prediction.astype(np.uint8))
image3D.CopyInformation(MRI)
save_path = os.path.join(save_dir,name)
sitk.WriteImage(image3D, save_path)