# SaNCE
This is a pytorch implementation of the SaNCE (SaNCE-TSR: A Temporal Point Process Framework for Right-in-time Sequential Recommendation.) and baseline methods

# Requeirements

- python 3.7.9
- pytorch 1.4.0
- pandas 1.1.2

# How to Run

1. preprocess the datasets:

   put the datasets in `data/origin_data/`

   run: `python main_preparedata.py` 

2. train our proposed model using `main_multi.py`

   run: `python main_multi.py`

   Three are some parameters in `main_multi.py`:

   - dataset: the dataset to run (`elec,movie_tv,home,sports,movielen`)

   - model:  the model name (`TimePredGRU4Rec, TimePredSASRec, GRU4Rec,SASRec,STAMP,SRGNN,GCSAN`)

   - train_method: the objective to optimize (`randome_ce,random_nce,neg_ce,neg_nce,adver_ce,adver_nce, ce,ce_time,mll`)

   - K: the number of  noise items

     