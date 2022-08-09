# Clustering_XMM_Newton

## Requirements
The pipeline uses hdbscan which can be installed as follows:
```
conda install -c conda-forge hdbscan
```
 
## How to run
```
git clone https://github.com/KaroFr/Clustering_XMM_Newton.git
cd Clustering_XMM_Newton
```
 
The following script gives an example on how the pipeline can be run on one observation. The observation ID can be changed in the script.
```
python main_one_obs.py
```
 
To run multiple observations in a row, run the following script. Again the observation IDs can be changed in the script.
```
python main_multiple_obs.py
```
