# deepAMPNet
The implementation of deepAMPNet for identification of antimicrobial peptides.
<p align="center">
	<img src="deepAMPNet.png"> 
</p>  

### Dependencies<br>
```
Python 3.7.0
torch 1.13.1
torch-cluster 1.6.0
torch-scatter 2.0.9
torch-sparse 0.6.15
torch-geometric 2.3.1
scikit-learn 1.0.2
biopython 1.81
h5py 3.8.0
numpy 1.21.6
pandas 1.3.5
```
More detailed python libraries used in this project are referred to ` requirements.txt ` , install the pytorch and pyG (torch-cluster, torch-scatter, torch-sparse, torch-geometric) according to your CUDA version.<br>
### Datasets<br>
Datasets for training and testing can be constructed using the method shown in 	`example/example.sh`, you are required to initially download the pre-trained Bi-LSTM protein language model from [here](http://bergerlab-downloads.csail.mit.edu/prose/saved_models.zip) and stored it in folder `Bi_LSTM_model`.The model we used is called `prose_mt_3x1024.sav`.<br>
```
python ../encode_AA.py -i example.fasta -o example.h5 -n 6165 -d -1
python ../dataset_h5.py --pdb example_pdb --label AMPs.txt --h5 example.h5 --threshold 20 --root dataset_example
```
### Train<br>
You can specify datasets to train and test model:  
```
python train.py --batch_size 64 --epochs 100 --learning_rate 0.0005 --train_dataset <your_train_dataset> --test_dataset <your_test_dataset> --dropout 0.5 --num_features 6165 --hidden_dim 512 --net HGPSL --save HGPSL_out
```
```
optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seed
  --batch_size BATCH_SIZE
                        batch size in training
  --epochs EPOCHS       number of epoch in training
  --learning_rate LEARNING_RATE
                        learning rate in training
  --train_dataset TRAIN_DATASET
                        Path of the train dataset
  --test_dataset TEST_DATASET
                        Path of the test dataset
  --save SAVE           path of saving output results
  --net {GCN,GAT,HGPSL}
                        GCN, GAT or HGPSL for model
  --dropout DROPOUT     dropout ratio for model
  --num_features NUM_FEATURES
                        number of input features
  --hidden_dim HIDDEN_DIM
                        number of hidden dimensions
  --aa_num AA_NUM       number of AA one-hot encoding
  --sample_neighbor SAMPLE_NEIGHBOR
                        whether sample neighbors for HGPSL Model
  --sparse_attention SPARSE_ATTENTION
                        whether use sparse attention for HGPSL Model
  --structure_learning STRUCTURE_LEARNING
                        whether perform structure learning for HGPSL Model
  --pool_ratio POOL_RATIO
                        pooling ratio for HGPSL Model
  --lamb LAMB           trade-off parameter for HGPSL Model
```
### Test&Predict<br>
You can specify datasets to test and use trained model to identify AMPs on your own data:  
```
python test.py --batch_size 32 --test_dataset <your_dataset_for_prediction> --model saved_model/deepAMPNet.pth --prefix out
```
```
optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size in testing
  --test_dataset TEST_DATASET
                        dataset of test data
  --model MODEL         path of model
  --prefix PREFIX       prefix of output csv
```
