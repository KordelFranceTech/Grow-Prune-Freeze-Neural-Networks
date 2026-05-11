import os
import sys

for i in range(0, 100):
    os.system(f"python3 visualize.py --agent gpf --config research/configs/exp11b_longer_episode.yaml --seed {i} --save trajectories/trajectory{i}.png")

#timeouts: 19,21,40,52,89,94