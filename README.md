# deepAMPNet
The implementation of deepAMPNet for identification of antimicrobial peptides.
<p align="center">
	<img src="deepAMPNet.png"> 
</p>
### Dependencies
* Python 3.7.0
* torch 1.13.1
* torch-cluster 1.6.0
* torch-scatter 2.0.9
* torch-sparse 0.6.15
* torch-geometric 2.3.1
* scikit-learn 1.0.2
* biopython 1.81
* h5py 3.8.0
* numpy 1.21.6
* pandas 1.3.5
More detailed python libraries used in this project are referred to `requirements.txt`, install the pytorch and pyG (torch-cluster, torch-scatter, torch-sparse, torch-geometric) according to your CUDA version.
### Datasets
Datasets for training and testing can be constructed using the method shown in 	`example/example.sh`, you are required to initially download the pre-trained Bi-LSTM protein language model from [here](http://bergerlab-downloads.csail.mit.edu/prose/saved_models.zip) and stored it in folder `Bi_LSTM_model`.The model we used is called `prose_mt_3x1024.sav`.
```
python ../encode_AA.py -i example.fasta -o example.h5 -n 6165 -d -1
python ../dataset_h5.py --pdb example_pdb --label AMPs.txt --h5 example.h5 --threshold 20 --root dataset_example
```
### Train
