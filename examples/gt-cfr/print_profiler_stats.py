import cProfile
import os
import pstats

def print_rlcard_function_stats(profiler):
    # Load the profiler into pstats
    stats = pstats.Stats(profiler)
    # Only include functions in the rlcard project
    project_root = os.path.dirname(os.path.abspath(__file__))
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename = func[0]
        if filename.startswith(project_root):
            stats.add_func_stats(func, (cc, nc, tt, ct, callers))
    # Simplify file paths for readibility
    stats.strip_dirs()
    # Sort by 'tottime percall' (average time per call)
    stats.sort_stats(pstats.SortKey.TIME)
    # Print the top 20 functions
    stats.print_stats(20)