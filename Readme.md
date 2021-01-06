### This is the repo which experiments the impact of pre-processing on classification of gender and age group from the a dataset of images of individuals which is obtained from https://paperswithcode.com/task/age-and-gender-classification

I completed this experiment in two parts:
1. Binary Classification for gender
2. A branched model to classify both gender and age group in the same network
Reference: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/



### 1. Model + Weight
1A. get_base_model.py -> This contains both the frozen pre-trained model + New layers on top used for classifying gender -> referred as primary model
1B. branched_model.py -> This is the branched model which is used for classifying both gender and age group-> referred as secondary model

#### NOTE Links for pre-trained weights not working 

#### 2. Training + Evaluation -> Primary Model
2A. Training.py -> Code for training primary model

2B. Training_Inference.ipynb -> Notebook for training

2C. Evaluation.py -> Code for model evaluation

#### 3. Training + Evaluation -> Secondary Model

3A branched_model.py -> Code for training secondary model

3B complete_training.py -> Code for training secondary model and evaluation


### 4. Steps and Reasoning behind Pre-processing and Data Balancing 

1. I observed that there were certain individuals who were present in both aligned and valid folders,
   so I decided to combine these and split them in such a way that no two similar individuals would be
   present in different splits.

2. In order to split data, I leveraged certain identifiers which were indicated by the first few characters,
   though there wasn't enough uniformity between them (Between different age groups).

3. Since some of my initial models sufferred from overfitting, one of the things I did to  counter that was use
   Data Augmentation (Blurring, Noise, Rotations, Flipping, Translations), some of the data in the truncated numpy
   arrays which I've added have those augmentations

4. Another observation from my initial models (Binary Classification) were the fact that boys under the age of 5 were being    identified as girls which was affecting metrics for that class, considering this was a simple binary classification problem and age range were not an important factor, so I simply randomly selected 500 boys from the range 1-5 and in order to maintain symmetry, I also selected girls from the same age range.

5. Since the predictions were almost balanced (even for earlier models), I concluded that the split was not causing any issues and subsequently primarily focused on reducing overfitting by augmentation or dropout.

6. Again for age split, data available did not have sufficient data for higher age groups so I resorted to data balancing by increasing range of age-groups for higher age and augmenting images and ensuring similar portion of ag-groups. There were 5 age groups which I divided the data into: 0-10, 11-20,21-35, 36-50, 50+


#### 5A. Observations - BInary

##### NOTE: Results added to results.txt for both models

1. Overfitting avoided by pre-processing as expected
2. Similar time for convergence maybe due to small scale of data?
3. Age balancing the dataset improves the overall accuracy of model


#### 5B. Observations - Branched Model

1. Accracy is again low for augmented data
2. Accuracy for 11-20 age group is low  -> Impact of puberty on facial features? -> Data Balncing yielded no positive results, changing age group only possible solution as of now.
3. Accuracy extremely high for lowest and highest age groups accross all the models
4. Convergence rate is almost similar 
5. Lower accuracy for 0-5 Age group again in gender classification


```python

```
