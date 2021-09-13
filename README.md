# DL-JHPF
Source codes of the article: P. Dong, H. Zhang and G. Y. Li, "Framework on Deep Learning-Based Joint Hybrid Processing for mmWave Massive MIMO Systems," IEEE Access, vol. 8, pp. 106023-106035, 2020. Please cite this paper when using the codes.

This folder contains codes for channel data generation executed in MATLAB and codes for channel estimation executed in Python.

1. Narrow band

   (1) Use MIMO_3GPP_channel_multi_fre.m to generate channel data for training and testing
   
   (2) Use DL_JHPF_train.py to train DL-JHPF and save model.
   
   (2) Use DL_JHPF_train_further.py to further train DL-JHPF based on the saved model.
   
   (3) Use DL_JHPF_test.py to test the performance of the trained DL-JHPF.
   
2. OFDM

   (1) Use MIMO_3GPP_channel_multi_fre.m to generate channel data for training and testing
   
   (2) Use DL_JHPF_ofdm_train.py to train DL-JHPF and save model.
   
   (2) Use DL_JHPF_ofdm_train_further.py to further train DL-JHPF based on the saved model.
   
   (3) Use DL_JHPF_ofdm_test.py to test the performance of the trained DL-JHPF.
