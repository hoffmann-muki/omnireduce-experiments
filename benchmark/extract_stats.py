#!/usr/bin/env python3
"""
Extract allreduce latency statistics from benchmark worker logs.
Computes per-worker averages, then overall min/max/avg/stddev.

Usage: python extract_stats.py <result_dir>
Outputs: node_count, msgsize, min, max, avg, stddev
"""
import sys
import os
import glob
import statistics

def extract_timings_from_log(logfile):
    """Extract 'time_only:VALUE;time_with_barrier:VALUE;' pairs from a worker log.
    Returns: (time_only_list, time_with_barrier_list)
    """
    time_only = []
    time_with_barrier = []
    try:
        with open(logfile, 'r') as f:
            for line in f:
                line = line.strip()
                if 'time_only:' in line and 'time_with_barrier:' in line:
                    # Parse both timers from: time_only:VALUE;time_with_barrier:VALUE;
                    import re
                    match = re.search(r'time_only:([0-9.e+-]+);time_with_barrier:([0-9.e+-]+);', line)
                    if match:
                        try:
                            time_only.append(float(match.group(1)))
                            time_with_barrier.append(float(match.group(2)))
                        except ValueError:
                            pass  # Skip malformed lines
    except FileNotFoundError:
        pass  # File doesn't exist yet
    return time_only, time_with_barrier

def compute_stats(values):
    """Compute min, max, avg, stddev for a list of values."""
    if not values:
        return None, None, None, None
    
    min_val = min(values)
    max_val = max(values)
    avg_val = statistics.mean(values)
    
    # Stddev (sample stddev if >1 value)
    if len(values) > 1:
        stddev_val = statistics.stdev(values)
    else:
        stddev_val = 0.0
    
    return min_val, max_val, avg_val, stddev_val

def extract_node_and_msgsize(result_dir):
    """Extract node_count and msgsize from result dir path."""
    # Expected format: results/node_<N>/msgsize_<SIZE>MiB
    parts = result_dir.strip('/').split('/')
    
    node_count = None
    msgsize = None
    
    for part in parts:
        if part.startswith('node_'):
            try:
                node_count = int(part.replace('node_', ''))
            except ValueError:
                pass
        elif part.startswith('msgsize_'):
            try:
                msgsize_str = part.replace('msgsize_', '').replace('MiB', '')
                msgsize = int(msgsize_str)
            except ValueError:
                pass
    
    return node_count, msgsize

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_stats.py <result_dir>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    # Find all worker logs
    worker_logs = sorted(glob.glob(os.path.join(result_dir, 'worker_*.log')))
    
    if not worker_logs:
        print("No worker logs found", file=sys.stderr)
        sys.exit(1)
    
    # Extract timings from each worker (now returns dual timers)
    all_time_only = []
    all_time_with_barrier = []
    
    for logfile in worker_logs:
        time_only, time_with_barrier = extract_timings_from_log(logfile)
        if time_only:
            all_time_only.extend(time_only)
        if time_with_barrier:
            all_time_with_barrier.extend(time_with_barrier)
    
    # Compute stats for both metrics
    stats_only = compute_stats(all_time_only) if all_time_only else (None, None, None, None)
    stats_barrier = compute_stats(all_time_with_barrier) if all_time_with_barrier else (None, None, None, None)
    
    # Extract node_count and msgsize from path
    node_count, msgsize = extract_node_and_msgsize(result_dir)
    
    # Output both metrics (format: node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_only_std,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg,time_with_barrier_std)
    if stats_only[0] is not None and stats_barrier[0] is not None:
        o_min, o_max, o_avg, o_std = stats_only
        b_min, b_max, b_avg, b_std = stats_barrier
        print(f"{node_count},{msgsize},{o_min:.1f},{o_max:.1f},{o_avg:.1f},{o_std:.1f},{b_min:.1f},{b_max:.1f},{b_avg:.1f},{b_std:.1f}")
    else:
        print("No timings extracted", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
