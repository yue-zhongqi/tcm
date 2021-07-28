# Implementation of Transporting Causal Mechanisms (TCM)

This includes the implementation of our ICCV2021 Oral paper [Transporting Causal Mechanisms for Unsupervised Domain Adaptation](https://arxiv.org/abs/2107.11055), where we give a theoretically-grounded solution to Unsupervised Domain Adaptation using the transportability theory, which requires the stratification and representation of the unobserved confounder. Specifically, we provide a practical implementation by identifying the stratification through learning Disentangled Causal Mechanisms (DCMs) and the representation through the proxy theory.

## Prerequisites

- pytorch = 1.0.1 
- torchvision = 0.2.1
- numpy = 1.17.2
- python3.6
- cuda10

## Preparation

1. Download datasets. For ImageCLEF-DA, please download the dataset using this link: https://drive.google.com/file/d/1_BXJlbalvW7I9xzHpMMy9k5SoCtQ3roJ/view?usp=sharing, where we organized the images in the dataset similar to how Office-Home is stored. Alternatively, you can download ImageCLEF-DA from official sources and process the dataset as given below. The other datasets can be downloaded from the official sources.

   ```
   imageclef_m
     |-- c (domain name)
       |-- 0 (class id)
       	|-- ...(images)
       |-- 1
       	|-- ...(images)
       ...
     |-- i
     	...
     |-- p
     	...
   ```

   

2. In the data folder, modify the file paths based on where dataset is stored. Note that the numbers following the file path is the class ID, and should not be modified. For example, for ImageCLEF-DA, change the file paths in data/ic/c.txt, data/ic/i.txt, data/ic/p.txt.

3. Under scripts/cyclegan, change *a_root* and *b_root* based on the dataset directory and domain name. Modify the *checpoints_dir* to where you want to store the trained DCMs.

4. The configs folder store the configurations of the trained DCMs. Modify these configuration files according to the dataset paths and checkpoint directories on your machine. Specifically, *a_root* and *b_root* are the paths to the stored dataset. *checkpoints_dir* is the saved DCMs networks location.  *cdm_path* is where the cross-domain counterfactual images will be saved (as a pre-processing step for the 2nd stage of TCM).

## Training and Testing

### Step 1: DCMs training

This corresponds to Section 3.1 of our paper.  The script to initialize training is stored in scripts/cyclegan. Run the python file to start DCMs training.

### Step 2: Generate Cross-Domain Counterfactual Images (\hat{X})

We pre-save the generated cross-domain counterfactual images for faster TCM training. This is achieved by running generate_cdm.py. However, this step is included as part of the automated scripts for the next step (corresponding to the *cdm* field in the configuration files in configs folder). So you don't need to worry about it.

### Step 3: Learning h_y and Inference

This is achieved by running the python files in scripts/tcm. The python scripts will first generate the counterfactual images, followed by training and testing.