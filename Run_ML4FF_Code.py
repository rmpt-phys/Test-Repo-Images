import os, time, sys, psutil

import pandas as pd

import multiprocessing

from copy import deepcopy

from datetime import datetime
from datetime import timedelta
 
import ML4FF_v7 as ml4ff

###########################
# --- Helper functions ---#
###########################

def log(msg):
    ##
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}")

def terminate_children():
    """
        Terminate all child processes of the current process.
    """
    try:
        parent = psutil.Process(os.getpid())

        for child in parent.children(recursive=True):
            ##
            log(f"Terminating child process PID {child.pid}"); child.terminate()

        gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=20)

        for p in alive:
            ##
            log(f"Forcing kill PID {p.pid}"); p.kill()

    except Exception as e:
        ##
        log(f"Error terminating children: {e}")

def get_columns(dataset_path):
    """
        Load CSV and select columns.
    """
    df = pd.read_csv(dataset_path).set_index("time-stamp")

    columns = ["time-stamp"]
    
    columns += [c for c in df.columns if c.startswith("PLU") or c.startswith("FLU")]

    return columns

def run_ml4ff_process(config_ml4ff, log_config):
    """
        Run ML4FF, write config.txt, and terminate children safely.
    """
    try:
        log(f"Running ML4FF PID {os.getpid()}")

        result = ml4ff.execute_ml4ff(config_ml4ff)

        # Append ML4FF CONFIG info

        log_config += ml4ff.CONFIG

        log_config.append("run_ML4FF: " + str(result))

        # Write config.txt

        config_file = os.path.join(config_ml4ff.result_path, 'config.txt')

        with open(config_file, 'w') as f:
            ##
            f.write('\n\n'.join(log_config))

        log(f"ML4FF finished. config.txt written.")

    except Exception as e:
        ##
        log(f"Error in ML4FF: {e}")

    finally:
        ##
        terminate_children()

        log(f"Process PID {os.getpid()} finished."); sys.exit(0)

def set_priority_recursive(pid, priority_class):
    """
        Set priority for a process and its children.
    """
    try:
        parent = psutil.Process(pid)
        ##
    except psutil.NoSuchProcess:
        ##
        return False
    
    all_procs = [parent] + parent.children(recursive=True)

    for p in all_procs:
        ##
        try:
            if p.nice() != priority_class:
                p.nice(priority_class)
            ####
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue

    return True        

#########################
# --- Main execution ---#
#########################

if __name__ == "__main__":

    #---------------------
    # Paths and parameters

    ALL_TRAIN = False

    DATASET_ROOT = 'ML4FF_Datasets/'

    RESULT_ROOT = f"ML4FF_Results/Run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}/"

    os.makedirs(RESULT_ROOT, exist_ok = True)

    DATASETS = ['stage413_and_rain-gauges_2015-2025_radar_compatible']

    LEAD_TIMES = ['10min', '60min', '120min', '240min']

    if ALL_TRAIN:
        ##
        ALGORITHMS =
            [             
            "sklearn_RandomForestRegressor",
            "sklearn_BaggingRegressor",
            "sklearn_GradientBoostingRegressor",
            "sklearn_LinearRegression",
            "sklearn_BayesianRidge",
            "sklearn_ARDRegression",
            "sklearn_LassoLars",
            "sklearn_LassoLarsIC",
            "sklearn_Lasso",
            "sklearn_ElasticNet",
            "sklearn_LassoCV",
            "sklearn_TransformedTargetRegressor",
            "sklearn_ExtraTreeRegressor",
            "sklearn_DecisionTreeRegressor",
            "sklearn_LinearSVR",
            "sklearn_PLSRegression",
            "sklearn_MLPRegressor",
            "sklearn_LGBMRegressor",
            "sklearn_DummyRegressor",
            "sklearn_TheilSenRegressor",
            "sklearn_OrthogonalMatchingPursuitCV",
            "sklearn_OrthogonalMatchingPursuit",
            "sklearn_RidgeCV",
            "sklearn_Ridge",
            "sklearn_SGDRegressor",
            "sklearn_PoissonRegressor",
            "sklearn_ElasticNetCV",
            "sklearn_KNeighborsRegressor",
            "sklearn_RadiusNeighborsRegressor",
            "sklearn_XGBRegressor",
            "sklearn_GaussianProcessRegressor",
            "sklearn_NuSVR",
            "DL_LSTM",
            "DL_LSTMPre"
            ]
    else:
        ALGORITHMS = [
                     'sklearn_XGBRegressor',
                     'sklearn_LinearSVR'
                     ]

    INNER_CV, OUTER_CV = 6,12

    HOLDOUT_SLICE = 0.875

    SEED = 42

    #------------------
    # Base ML4FF config

    base_config = ml4ff.Config(
        save_models=True,
        inner_cv=INNER_CV,
        outer_cv=OUTER_CV,
        holdout_slice=HOLDOUT_SLICE,
        seed=SEED,
        dataset_path="",
        dataset_columns=[],
        result_path="",
        ml_algorithms=[],
        dl_algorithms=[]
    )

    #-----------------------------------
    # CPU affinity and priority settings

    CPU_AFFINITY_QTD = 2 # set None to skip

    if CPU_AFFINITY_QTD is not None:
        #.....................................................#
        try:
            parent = psutil.Process(os.getpid())

            allowed_cpus = list(range(CPU_AFFINITY_QTD))

            parent.cpu_affinity(allowed_cpus)

            log(f"- Parent CPU affinity set: {allowed_cpus}\n")
            ##
        except Exception as e:
            ##
            log(f"Could not set parent CPU affinity: {e}\n")
        #.....................................................#
        #
    else: log(f"CPU affinity control is disabled;\n")
        
    # ---------------
    # Experiment loop

    CONFIG = [] # keep global log

    CONFIG.append(f"SEED: {SEED}")
    CONFIG.append(f"INNER_CV x OUTER_CV: {INNER_CV} x {OUTER_CV}")
    CONFIG.append(f"HOLDOUT_SLICE: {HOLDOUT_SLICE}")

    #--------------------------
    # Loop over all experiments

    start_time = time.time()
    
    for dataset in DATASETS:
        ##
        for lead_time in LEAD_TIMES:
            ##
            for algo in ALGORITHMS:
                ##
                ## Prepare dataset and columns

                csv_path = os.path.join(DATASET_ROOT, f"{dataset}-{lead_time}.csv")

                columns = get_columns(csv_path)

                ## Prepare result folder

                subfolder = f"{algo}/{dataset}-{lead_time}"

                result_path = os.path.join(RESULT_ROOT, subfolder)

                os.makedirs(result_path, exist_ok=True)

                ## Copy config for this run

                config = deepcopy(base_config)

                config.dataset_path = csv_path

                config.dataset_columns = columns

                config.result_path = result_path

                config.ml_algorithms = [algo]

                config.dl_algorithms = []

                ## Print info. for user

                log(f"> Starting process:")                
                log(f"- Algorithm: {algo}")
                log(f"- Dataset: {dataset}")
                log(f"- LeadTime: {lead_time}")
                log(f"- InnerCV: {INNER_CV}")
                log(f"- OuterCV: {OUTER_CV}")
                log(f"- HoldOut: {HOLDOUT_SLICE}")
                log(f"- Dataset path: {csv_path}")
                log(f"- Result root: {RESULT_ROOT}")
                log(f"- Result path: {subfolder}")
                log(f"- CPU Affinity count: {CPU_AFFINITY_QTD}\n")

                ## Launch process and control execution CPU affinity

                p = multiprocessing.Process(target=run_ml4ff_process,
                                            args=(config, deepcopy(CONFIG)))
                p.start()

                if CPU_AFFINITY_QTD is not None:
                    ##
                    try:
                        ps_proc = psutil.Process(p.pid)

                        ps_proc.cpu_affinity(allowed_cpus)

                        log(f"Child PID {p.pid} CPU affinity set: {allowed_cpus}")

                        set_priority_recursive(p.pid, 10)
                        ##
                    except Exception as e:
                        ##
                        log(f"Could not set child CPU affinity: {e}")

                p.join()

                elapsed = timedelta(seconds=round(time.time() - start_time)) 

                log("Process finished. Waiting 5 seconds...")

                time.sleep(5)

                ##( END-LOOPS )
                
    log(f"All processes finished in {elapsed} (hh:mm:ss)\n")
    #
    ## END main execution ( if __name__ == "__main__" )
