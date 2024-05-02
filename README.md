# Overview:
This repository contains the not-as-clean-as-I-would-like code to reproduce our submission to the justRAIGS competition 
on [justifiable glaucoma screening prediction](https://justraigs.grand-challenge.org/) from eye fundus images.
The data needed to train all models lives in [zenodo](https://zenodo.org/uploads/10035093), and needs to be pre-processed 
first - standard things like cropping to FOV and discarding terrible images. 
This is done by calling to the `preprocess_training_data.py` script, providing the input and output folder as parameters. 

Once the imaging part of the data is in-place, we also need to manipulate labels. In fact, this is core to our solution, 
which consists on combining information from multiple annotators of different skills into a tailored label smoothing scheme 
that allows us to leverage the available large collection of fundus images, instead of simply discarding samples with inter-rater variability. 
There are two sub-tasks in our problem, glaucoma screening and glaucoma justification. The way we manipulate labels for each case is explained 
in our short paper ([here](https://github.com/agaldran/justraigs/blob/main/paper/ISBI24_paper_1635.pdf)) and illustrated in two notebooks, `data_splitting_rg.ipynb` and `data_splitting_features.ipynb`. 
After understanding our strategy, you can use the scripts `split_training_data.py` and `split_training_data_features.py` to apply it and generate 
several `.csv` files that you will need for five-fold model training. The logic to handle these labels is also partially implemented in `utils/data_load.py`.

Once all the above is done, we can train the several models used in our solution by means of the `train.py` and `train_features.py` scripts. 
The exact calls needed for this are contained in a couple of bash scripts<sup>[1](#myfootnote1)</sup>: `run_justraigs_experiments_rg.sh` and `run_justraigs_experiments_just.sh`. 
Additionally, there is a `utils` folder which contains mostly auxiliary logic for data loading, model definitions, metrics, etc.
Any doubt please open an issue.

**Note**: There is also [this Grand-Challenge algorithm](https://grand-challenge.org/algorithms/data-centric-justraigs/) that contains our exact submission to the contest.

---

<a name="myfootnote1">1</a>: The solution of course ended up being a heavy ensembling. The glaucoma screening side had a combination of efficientB2 and efficientB3 models, 
and the justification part had a combination of resnet50, efficientB0, efficientB1 and efficientB2 models. 
TTA is also applied, and a trick to predict justification on images with low glaucoma likelihood using only one model instead of the full ensemble. 
For reference, I am also adding the docker container generation code in the `docker` folder, check out the `inference.py` file there if you are interested.


