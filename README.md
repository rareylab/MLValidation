# Evaluation of Convolutional Neural Networks for Structure-based Virtual Screening
The here provided files are part of the publication: 

**Sieg. J., Flachsenberg F., Rarey M. _In The Need of Bias Control: Evaluating Chemical Data for Machine Learning in 
Structure-Based Virtual Screening. 2018_**

We provide the code mentioned in the paper for the two reimplementations of
published Convolutional Neural networks (CNN) for the scoring of docked 
protein-ligand complexes in Virtual Screening. 

The goal of the paper was to evaluate variations of the descriptors 
and featurization of the input. Contrary to the original networks the reimplemented CNNs are not based on molecular
docking. Instead of using the protein structure in complex with a small molecule only small molecules are used 
while the protein is excluded completely. These networks are therefore ligand-based
versions of the orginals. Still similar prediction performance
can be achieved. Details are shown and discussed in the paper.

## Authors and License

Developed at the [University of Hamburg](https://www.uni-hamburg.de/), 
[ZBH - Center for Bioinformatics](http://www.zbh.uni-hamburg.de), 
[Research Group for Computational Molecular Design](http://www.zbh.uni-hamburg.de/amd) by Jochen Sieg, Florian Flachsenberg and Matthias Rarey.

The files are distributed under New BSD license,
see the file LICENSE and [New BSD](https://opensource.org/licenses/BSD-3-Clause). 

## Prerequisites

The scripts are written using Python3.6 and different packages:

```rdkit, sklearn, tensorflow, keras, imbalanced-learn  ```

For example install with conda:

```
conda install rdkit scikit-learn tensorflow keras
conda install -c conda-forge imbalanced-learn 
 ```

The data used for evaluation are the [DUD](dud.docking.org) (Partial Charges recalculated by Inhibox)  and [DUD-E](dude.docking.org) datasets.

## 1. Reimplemented CNN: DeepVS
DeepVS is a CNN inspired from natural language processing. The original publication:

_Pereira, J.C., Caffarena, E.R., dos Santos, C.N.
Boosting Docking-Based Virtual Screening with Deep Learning.
J.Chem.Inf.Model., 2016, 56 (12), 5495-2506,
DOI: 10.1021/acs.jcim.6b00355_

Our reimplementation of the CNN can be used by first calculating descriptors for the input molecules and then
training and evaluating the network:

```
python deepvs_descriptor.py --input allDUDfiles_AM1/*.mol2 --output descriptor.csv --reset
```

Train and evaluate the network. test_id specifies the protein target to use as 
test set in the leave-one-out cross validation:

```
python DeepVS.py --input descriptor.csv --test_id 0 --exclude excludes.json
```


## 2. Reimplemented CNN: Grid-based 3D-CNN

The original version of this network is based on a 3D-grid centered around 
the binding site and can be found in this publication:

_Ragoza, M., Hochuli, J., Idrobo, E., Sunseri, J., Koes D.R.
Protein–Ligand Scoring with Convolutional Neural Networks.
J.Chem.Inf.Model., 2017, 57 (4), 942-957,
DOI: 10.1021/acs.jcim.6b00740_

The original code from _Ragoza et al._ can be found at [gnina](https://github.com/gnina).


Our network only reads in molecules (no protein), and puts the molecules into
a grid for featurization. 

Calculate descriptors for all molecules:

```
python grid_descriptor.py --input DUD_E/*/*.smi --output grid3Dmols
```


Train and evaluate the network:
```
python grid_cnn.py --input grid3Dmols --folds folds.json
```

The cross validation folds specified in folds.json have been created with [clustering.py](https://github.com/gnina/scripts/blob/9cf8892010873b672f370a122e32aa8bc496a5e1/clustering.py):
