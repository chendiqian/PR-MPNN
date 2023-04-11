## unofficial one to remind myself how to execute experiments

for fixed hyperparams, run e.g. 
`python main_fixconf.py with configs/zinc/global/topk20_1_random.yaml`

for sweep, run
`wandb sweep configs/zinc/global/sweep_20_1_simple.yaml`
`wandb agent $ID`