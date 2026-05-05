from collections import defaultdict
import logging
import numpy as np
import wandb


class Logger:
    def __init__(self, console_logger, notes="", tags=[], group=""):
        wandb.login()
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

        # Define a set of metrics that should be treated as raw integer values
        self.integer_metrics = {
            "goals_reached", "test_goals_reached", 
            "max_possible_goals", "test_max_possible_goals",
            "total_goals_reached", "test_total_goals_reached",
            "total_episodes", "test_total_episodes"
        }

        wandb.init(project="6756-rl-experiments", notes=notes, tags=tags, group=group)

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        wandb.log({key: value}, step=t)
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)


    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            
            # Special case for goals_reached - use most recent interval value
            if k in self.integer_metrics:
                if len(self.stats[k]) > 0:
                    # Get just the most recent value
                    try:
                        most_recent_goals = self.stats[k][-1][1]
                        item = "{:.0f}".format(most_recent_goals)  # Format as integer
                    except:
                        most_recent_goals = self.stats[k][-1][1].item()
                        item = "{:.0f}".format(most_recent_goals)  # Format as integer
                else:
                    item = "0"
            else:
                # Regular windowed average for other metrics
                window = 5 if k != "epsilon" else 1
                try:
                    item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
                except:
                    item = "{:.4f}".format(
                        np.mean([x[1].item() for x in self.stats[k][-window:]])
                    )
            
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)        

# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
