# Token ID definitions — single source of truth.
# NEVER reorder existing entries; only append new ones to preserve checkpoint compatibility.

# Left concentration stream (7 tokens)
L_BLANK = 0
L_C0 = 1
L_C1 = 2
L_C2 = 3
L_C3 = 4
L_C4 = 5
L_C5 = 6

# Right concentration stream (7 tokens)
R_BLANK = 0
R_C0 = 1
R_C1 = 2
R_C2 = 3
R_C3 = 4
R_C4 = 5
R_C5 = 6

# Wind direction stream in agent frame (8 octants, 0 = headwind, increasing clockwise)
W_0 = 0   # directly ahead (headwind)
W_1 = 1
W_2 = 2
W_3 = 3   # directly behind (tailwind)
W_4 = 4
W_5 = 5
W_6 = 6
W_7 = 7

# Action stream (6 tokens)
A_FWD = 0
A_LEFT15 = 1
A_RIGHT15 = 2
A_TURN180 = 3
A_CAST_L = 4
A_CAST_R = 5

# Special tokens
PAD = 0
BOS = 1
EOS = 2
RESET = 3

N_CONC_BINS = 7    # BLANK + C0..C5
N_WIND_OCTANTS = 8
N_ACTIONS = 6

# TODO: add action of 'wait' or 'do nothing'
ACTION_NAMES = ['forward', 'turn_left_15', 'turn_right_15', 'turn_around', 'cast_left', 'cast_right']
