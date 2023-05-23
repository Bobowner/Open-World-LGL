# Open-World-LGL

The code to the paper Open-World-Lifelong Graph Learning.

## Structure
The start point of the code is main.py. 
The dataset is constructed in load_dataset.py and data_iterators.py.
In pre_compute.py the powers of A for graph-mlp can be precomputed.
The files built_exp_setting.py, handle_meta_data.py and results_writer.py are to manage the experiments.
The files experiment.py and evaluation.py are to perform the actual experiment. 
Models can be found in the models folder or the ood_models folder.

## Run the Code
To run an experiment, define a .yml file in the "experiments" folder and run "main.py --experiment experiments/dummy.yml", where dummy.yml is your yamel file.
You can find possible parameters for you experiment in the folder "default parameters".
If you do not set a specific value for an experiment, the value is set to the provided value in "default parameters".
If you provide multiple values for a parameter, the code will a an experiment for each possible value combination in the yaml file.
