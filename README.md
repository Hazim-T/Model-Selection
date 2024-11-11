# Transfer Learning with Limited Data for COVID-19 Chest X-Ray Classification

## Overview

This project demonstrates the application of **transfer learning** in **limited data scenarios, in the 
medical field**. Data collection in medical fields has been scarce and difficult because of variations in
patient data and imaging techniques, in addition to legal restrictions and many other reasons, so the use
of transfer learning to analyze patterns even with very limited data is crucial.

To demonstrate, we are going to use the **COVID-19 Chest X-ray Dataset**, shrunk down to contain only 
**150 training images**. We focus on the importance of choosing the correct model for transfer learning
which varies based on the type of data and the application, as in it is case specific.

## Objectives

1.  To perform and observe the performance of transfer learning on a small COVID-19 Chest X-ray Dataset.
2. **Some pre-trained models comparison** to find the best performer.

## Dataset

The dataset used here has **chest X-ray images** to classify whether a patient is suffering from 
COVID-19 or not, in addition to viral pneumonia which can be caused by Covid-19 but is difficult to
differentiate in x-ray scans. 

## Models Used with Transfer Learning

Pre-trained models tried:

* **ResNet50**
* **VGG16**
* **InceptionV3**
* **MobileNetV2**

All these models were trained on imagenet. Those models were attached to a classifier and fine-tuned on the 
dataset for further improving their performance.

## Key Observations

- The variance of model performances indicates that **the choice of model is very critical**, specially when 
dealing with limited data.


- DenseNet and VGG16 were the best performers on our dataset, while being very simple to fine tune.


- ResNet50 was the worst performer, although it was not fine-tuned due to the complexity of tuning it.


- If a model does not perform well on this dataset, that does not mean the model is bad, it simply means 
that the dataset is not suitable for the model.

## Conclusion

This work thus shows that transfer learning has endless applications and is crucial to creating a simpler
development environment for AI progress. Experimenting with multiple models when choosing a model is 
extremely important for better results, as well as providing a varying level of knowledge needed to work 
with the model as some are more complex to work with.