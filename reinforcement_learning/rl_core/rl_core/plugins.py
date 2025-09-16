from importlib.metadata import entry_points

# ANSI colors (avoid circular import)
RED = "\033[31m"
RESET = "\033[0m"

def load_task_spec(task_name: str):
    eps = entry_points(group='rl_tasks')
    matches = [ep for ep in eps if ep.name == task_name]
    if not matches:
        raise RuntimeError(f"{RED}Task '{task_name}' not found in entry_points 'rl_tasks'{RESET}")
    return matches[0].load()()  # instantiate