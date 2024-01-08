python ../encode_AA.py -i example.fasta -o example.h5 -n 6165 -d -1
python ../dataset_h5.py --pdb example_pdb --label AMPs.txt --h5 example.h5 --threshold 20 --root dataset_example
