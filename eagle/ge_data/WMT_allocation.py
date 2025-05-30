import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = 0
# 더 작은 수로 시작하여 테스트
e = 30000 - 1  # 필요에 따라 조정 가능
gpus=[[0],[1],[2],[3],[4],[5],[6],[7]]
num_p = len(gpus)
outdir = '{}/wmt16_en_de_{}_{}_mufp16'.format(args.outdir,s,e)


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    command = "python /home/chokwans99/LoRA_Eagle/EAGLE_LoRA/eagle/ge_data/WMT_ge_vicuna.py --start={} --end={} --index={} --gpu_index {} --outdir {}".format(start, end, index,
                                                                                                gpu_index_str, outdir)
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
