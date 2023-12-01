# HGW-TC-Experimental-code
 The code realization of comparative experiment in paper “Network Traffic Classification based on Edge Gateway using Deep Federated Learning”.
# The operating environment is as follows:
  tensorflow-gpu 2.6.0 cudatoolkit 11.3.1  cudnn 8.2.1   pandas 2.1.2   numpy  1.23.4  

# Folder description:
  Exp2 is the comparison of semi-supervised learning.
  Exp3 is a comparison between federal AECNN and federal baseline CNN.
  Exp4 is a comparison of several semi-supervised federal learning methods.

# Description of part of the code
  getlabelindex(Y_full,n_classes,labelnum)  This method randomly selects labelnum tags by category from the Y_full. 
  This method is used to split tagged data
  Labelnum can control the quantity of labeled data for each category （But they are no use in FFSL.py and FLUIDS.py）
  
  ProposedFLAECNN.py differs from FLUIDS.py and FFSL.py in:
  ProposedFLAECNN.py conducts unsupervised and supervised training in every "for clientID in range (numOfClients):"
  FLUIDS.py and FFSL.py do supervised training in every "for iterationNo in range (1 for clientID in range (numOfClients):") and unsupervised training in each "OfIteration (training):"

  This simulates the ability of sub nodes to have data labeling, so ProposedFLAECNN can perform supervised training on each child node

