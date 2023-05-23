from handle_meta_data import find_pars_yaml, load_yml, get_combinations_of_yml,fill_defaults, delete_last_setting
from built_exp_setting import setup_experiment
from experiment import run_one_setting
from results_writer import create_setting_folder
from tqdm import tqdm
import multiprocessing
import argparse

#TODO: surpress warings of no class sklearn metrics
def f(idx,exp):
    
    setting = setup_experiment(exp)
    create_setting_folder(setting, idx)
    run_one_setting(setting, idx)
    return 0


def main(ARGS):
    
    all_yaml_paths = find_pars_yaml(ARGS)
    
    
    for path in all_yaml_paths:
        dic = load_yml(path)
        
        all_exps = get_combinations_of_yml(dic)
        
        offset = 0
        if ARGS.proceed:
            
            setting_number = delete_last_setting(ARGS)
            all_exps = all_exps[setting_number:]
            offset = setting_number
        
        for idx, exp in tqdm(enumerate(all_exps)):
            print(exp)

            setting = setup_experiment(exp)
            create_setting_folder(setting, idx+offset)
            print("\nProcess setting: " + str(idx+offset+1)+"/"+str(len(all_exps)+offset)+"\n")
            run_one_setting(setting, idx+offset)

            
parser = argparse.ArgumentParser()
parser.add_argument('--experiment',nargs='+', default=None, help="Pass paths of your experiment yaml files")
parser.add_argument('--proceed', action='store_true', default=False, help="Set true to continue experiment")
ARGS = parser.parse_args()

main(ARGS)