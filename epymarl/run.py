import os
import sys
import datetime
import pprint
import random
import time
import threading
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath, join
import glob

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


# ==================== SLURM + GPU OPTIMIZATION ====================
def setup_slurm_gpu_environment():
    """Optimize environment for SLURM GPU jobs"""
    print("=" * 60)
    print("SLURM GPU SETUP")
    print("=" * 60)
    
    # 1. SLURM GPU Detection
    slurm_gpus = os.environ.get('SLURM_GPUS', '0')
    slurm_gpus_on_node = os.environ.get('SLURM_GPUS_ON_NODE', '0')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    print(f"SLURM_GPUS: {slurm_gpus}")
    print(f"SLURM_GPUS_ON_NODE: {slurm_gpus_on_node}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 2. Force CUDA visibility if not set
    if not cuda_visible_devices:
        if slurm_gpus_on_node and slurm_gpus_on_node != '0':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("SET: CUDA_VISIBLE_DEVICES=0 (from SLURM)")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("SET: CUDA_VISIBLE_DEVICES=0 (default)")
    
    # 3. CPU Optimization for SLURM
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '1')
    slurm_ntasks = os.environ.get('SLURM_NTASKS', '1')
    
    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    print(f"SLURM_NTASKS: {slurm_ntasks}")
    
    # Set optimal CPU threads
    optimal_threads = min(int(slurm_cpus), 16)  # Cap at 16 for stability
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    
    print(f"SET: OMP_NUM_THREADS={optimal_threads}")
    print("=" * 60)
    
    return optimal_threads

# Call SLURM setup BEFORE any imports
optimal_threads = setup_slurm_gpu_environment()




def setup_gpu_optimization():
    """Advanced GPU optimization for training"""
    if not th.cuda.is_available():
        return False
    
    print("=" * 60)
    print("GPU OPTIMIZATION SETUP")
    print("=" * 60)
    
    # 1. GPU Information
    gpu_count = th.cuda.device_count()
    for i in range(gpu_count):
        gpu_name = th.cuda.get_device_name(i)
        gpu_memory = th.cuda.get_device_properties(i).total_memory
        print(f"GPU {i}: {gpu_name} ({gpu_memory/1024**3:.1f}GB)")
    
    # 2. GPU Performance Optimizations
    th.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    th.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    th.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
    th.backends.cudnn.allow_tf32 = True  # Enable TF32 for cudnn
    
    # # Set matrix multiplication precision for even better performance
    # try:
    #     th.set_float32_matmul_precision('medium')  # Options: 'highest', 'high', 'medium'
    #     print("Set float32 matmul precision to 'medium' for better performance")
    # except AttributeError:
    #     print("torch.set_float32_matmul_precision not available (PyTorch < 2.0)")
    # except Exception as e:
    #     print(f"Could not set matmul precision: {e}")
    
    # 3. Memory Management
    th.cuda.empty_cache()
    
    # 4. Set memory fraction (use 90% of GPU memory)
    try:
        total_memory = th.cuda.get_device_properties(0).total_memory
        memory_fraction = 0.9
        th.cuda.set_per_process_memory_fraction(memory_fraction)
        print(f"GPU Memory Fraction: {memory_fraction} ({total_memory*memory_fraction/1024**3:.1f}GB)")
    except:
        print("Could not set memory fraction")
    
    print("GPU optimizations enabled")
    print("=" * 60)
    return True


def get_next_run_number(base_path, seed, name):
    pattern = f"{name}_seed{seed}_*"
    existing_runs = glob.glob(join(base_path, pattern))
    return len(existing_runs) + 1


def run(_run, _config, _log):
    print("=" * 80)
    print("STARTING SLURM GPU-OPTIMIZED TRAINING")
    print("=" * 80)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or job_id.strip() == "":
        job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if "env_args" not in _config or _config["env_args"] is None:
        _config["env_args"] = {}
    _config["env_args"]["job_id"] = job_id
    _config["job_id"] = job_id

    print(f"[Runner] Using job_id: {job_id}")
    
    # 1. Seed Configuration
    if "seed" not in _config or _config["seed"] is None:
        _log.warning("No seed specified in config. Using default seed 123")
        _config["seed"] = 123  
    _config["seed"] = int(_config["seed"])

    # 2. GPU Setup
    gpu_available = setup_gpu_optimization()
    _config["use_cuda"] = gpu_available
    
    # 3. Set random seeds
    _log.info(f"Setting random seed to {_config['seed']}")
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    
    if gpu_available:
        th.cuda.manual_seed_all(_config["seed"])
    
    # 4. PyTorch Threading Optimization
    th.set_num_threads(optimal_threads)
    _log.info(f"PyTorch threads set to: {th.get_num_threads()}")
    
    # 5. Device Selection (moved before args creation)
    device = "cuda" if gpu_available else "cpu"
    _log.info(f"Using device: {device}")
    
    # 6. Environment seed
    if "env_args" not in _config:
        _config["env_args"] = {}
    _config["env_args"]["seed"] = _config["seed"]

    # 7. Determine Sacred tracking directory FIRST
    sacred_dir = _run.observers[0].dir
    tracking_dir = os.path.join(sacred_dir, "action_logs")
    os.makedirs(tracking_dir, exist_ok=True)
    print(f"[Tracking] All logs will go in: {tracking_dir}")

    # Pass tracking_dir to env_args BEFORE creating args
    if "env_args" not in _config:
        _config["env_args"] = {}
    _config["env_args"]["tracking_dir"] = tracking_dir

    # 8. NOW do sanity check and create args object
    _config = args_sanity_check(_config, _log)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    # 9. Setup logging
    logger = Logger(_log, _config["note"], _config["tags"])
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")
    
    # 10. Unique token and paths
    try:
        map_name = _config["env_args"]["map_name"]
        unique_token = (
            f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except:
        map_name = _config["env_args"].get("key", "default")
        unique_token = (
            f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    results_path = join(dirname(dirname(abspath(__file__))), "results", "models")
    run_number = get_next_run_number(results_path, _config['seed'], _config['name'])
    
    args.unique_token = unique_token
    
    # 11. Tensorboard setup
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    
    # 12. Log experiment info
    with open("seed_log.txt", "a") as f:
        f.write(f"Run: {_config['name']} - Seed: {_config['seed']} - Device: {args.device} - RunNumber: {run_number}\n")
    print(f"Experiment Seed Logged: {_config['seed']} - Run Number: {run_number}")
    
    logger.setup_sacred(_run)
    
    # 13. Run training
    run_sequential(args=args, logger=logger)
    
    # 14. Cleanup
    print("=" * 80)
    print("TRAINING COMPLETED - CLEANING UP")
    print("=" * 80)
    
    if th.cuda.is_available():
        th.cuda.empty_cache()
        print("GPU cache cleared")
    
    # Thread cleanup
    for t in threading.enumerate():
        if t.name != "MainThread":
            print(f"Cleaning up thread: {t.name}")
            if hasattr(t, 'join'):
                t.join(timeout=1)
    
    print("Cleanup completed")


def run_sequential(args, logger):
    """Optimized sequential training with GPU memory management"""
    
    # 1. Initial GPU status
    if args.use_cuda:
        print("=" * 60)
        print("INITIAL GPU MEMORY STATUS")
        print("=" * 60)
        print(f"Allocated: {th.cuda.memory_allocated(0)/1024**2:.1f}MB")
        print(f"Reserved: {th.cuda.memory_reserved(0)/1024**2:.1f}MB")
        print(f"Free: {th.cuda.memory_reserved(0)/1024**2 - th.cuda.memory_allocated(0)/1024**2:.1f}MB")
        print("=" * 60)
    
    # 2. Initialize runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    
    # 3. Environment setup
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"] 
    
    # 4. Buffer setup with GPU optimization
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "roles": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "role_avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    # Use GPU for buffer if available and memory allows
    buffer_device = args.device if args.use_cuda and not args.buffer_cpu_only else "cpu"
    
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device=buffer_device,
    )
    
    # 5. Controller and learner setup
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    
    if args.use_cuda:
        learner.cuda()
        logger.console_logger.info(f"Learner device: {next(learner.mac.parameters()).device}")
    
    # 6. Checkpoint loading
    if args.checkpoint_path != "":
        timesteps = []
        if os.path.isdir(args.checkpoint_path):
            for name in os.listdir(args.checkpoint_path):
                full_name = os.path.join(args.checkpoint_path, name)
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))
            
            if timesteps:
                timestep_to_load = max(timesteps) if args.load_step == 0 else min(timesteps, key=lambda x: abs(x - args.load_step))
                model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
                logger.console_logger.info(f"Loading model from {model_path}")
                learner.load_models(model_path)
                runner.t_env = timestep_to_load
                
                if args.evaluate or args.save_replay:
                    evaluate_sequential(args, runner)
                    return
    
    # 7. Training loop with GPU optimization
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    start_time = time.time()
    last_time = start_time
    
    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")
    
    # GPU memory monitoring intervals
    memory_check_interval = args.log_interval
    memory_cleanup_interval = args.log_interval * 5
    
    while runner.t_env <= args.t_max:
        
        # GPU memory monitoring
        if args.use_cuda and (runner.t_env % memory_check_interval == 0) and runner.t_env > 0:
            allocated = th.cuda.memory_allocated(0) / 1024**2
            reserved = th.cuda.memory_reserved(0) / 1024**2
            
            if runner.t_env % (memory_check_interval * 10) == 0:
                logger.console_logger.info(f"Step {runner.t_env}: GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
            
            # Aggressive cleanup if memory usage is high
            if allocated > 8000:  # > 8GB
                th.cuda.empty_cache()
                logger.console_logger.info(f"GPU memory cleanup triggered at {allocated:.1f}MB")
        
        # Regular cleanup
        if args.use_cuda and (runner.t_env % memory_cleanup_interval == 0) and runner.t_env > 0:
            th.cuda.empty_cache()
        
        # Training step
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            
            # Ensure data is on correct device
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            
            learner.train(episode_sample, runner.t_env, episode)
            
            # Clean up episode sample
            del episode_sample
        
        # Testing
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            logger.console_logger.info(
                f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.t_max)}, "
                f"Time passed: {time_str(time.time() - start_time)}"
            )
            
            last_time = time.time()
            last_test_T = runner.t_env
            
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
        
        # Update best actions
        runner.update_best_actions()
        runner.update_best_test_actions()
        
        # Model saving
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")
            
            learner.save_models(save_path)
            runner.save_best_actions(save_path)
            runner.save_best_test_actions(save_path)
        
        episode += args.batch_size_run
        
        # Logging
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    # Final cleanup
    if args.use_cuda:
        final_allocated = th.cuda.memory_allocated(0) / 1024**2
        final_reserved = th.cuda.memory_reserved(0) / 1024**2
        logger.console_logger.info(f"Final GPU Memory - Allocated: {final_allocated:.1f}MB, Reserved: {final_reserved:.1f}MB")
        th.cuda.empty_cache()
    
    runner.close_env()
    logger.console_logger.info("Training completed successfully")


def evaluate_sequential(args, runner):
    """Evaluation with GPU optimization"""
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)
    
    if args.save_replay:
        runner.save_replay()
    
    runner.close_env()


def args_sanity_check(config, _log):
    """Enhanced sanity check with GPU validation"""
    
    # CUDA availability check
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF - no CUDA devices available!")
    
    # GPU memory check
    if config["use_cuda"]:
        try:
            total_memory = th.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory < 4:  # Less than 4GB
                _log.warning(f"GPU has only {total_memory:.1f}GB memory - consider reducing batch size")
        except:
            pass
    
    # Test episode adjustment
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
    
    return config