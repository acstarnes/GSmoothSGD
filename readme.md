# Readme:

The following are the programs contained in the Supplemental Material:
1. `hyperparameter_search.py`:
    - This can be used to perform a hyperparameter search for GSmoothSGD and GSmoothAdam.
    - Using the unsmooth cuild cnn function allows a hyperparameter search to be done on SGD and Adam.
2. `noise_heatmap_single_sigma.py`:
    - This can be used to generate the data for experiments on GSmoothSGD and GSmoothAdam.
    - Visualization was done with `sns.heatmap` (after using `import seaborn as sns`)
3. `svrg_noise_heatmaps.py`:
    - This can be used to generate the data for the GSmoothSVRG experiments.
    - Many csv files will be generated (one for each method and experiment done)
    - Visualization was done with `sns.heatmap` (after using `import seaborn as sns`)