# Grow-Prune-Freeze Networks

### Adaptive and continual deep learning grounded in random matrix theory.
----

This repository is associated with the paper _Grow-Prune-Freeze Networks: Adaptive & Continual Learning Technique for Olfactory Navigation_ [1](https://arxiv.org/abs/2605.25170) [2](https://openreview.net/pdf?id=TPOM2VTfau) by Kordel France and Ovidiu Daescu.
We present an early idea for a deep learning model that continually trains itself, adds its own layers, prunes low-rank neurons, and freezes key temporal knowledge.
Our primary motivation for _GPFs_ was driven from a desire to have generalizable reinforcement learning models for olfactory navigation that could quickly adapt to the chemical plume's complexity.

The repo contains the code and experiments associated with the main body and appendices of the paper.

`visualize.py` runs everything needed to produce an _experiment_ located in the `./research` folder.
The `./research/configs` folder contains previously run experiments used to acquire the optimal GPF architecture for the final model discussed in the main body of the paper.
We recommend using Karpathy's [_autoresearch_](https://github.com/karpathy/autoresearch) concept to continually run experiments and tune GPFs for other tasks outside of those discussed in the paper.  
Claude Code was used to continually run experiments in this fashion and the results of this can be found in the `./research/` and `./checkpoints` subdirs.
The rest of the repository is structured as follows:


```
GPF-Olfactory-Nav/
├── agent/                                  # Construction of the RL agent
├── checkpoints/                            # Model checkpoint of the best performer
├── configs/                                # Config files for the RL agent and world model
├── gpf/                                    # Training code
├── simulator/                              # Construction of the training environment
├── tokenizer/                              # Tokenizing the plume
├── trajectories/                           # Saved images from experimental runs
├── validation/                             # Evaluate the agent's performance
└── visualization/                          # Scripts for visuzalizing agent performance
```

If you leverage _Grow-Prune-Freeze Networks_ and/or this repository in your own work, please be sure to cite it with the following:


```
@misc{france2026growprunefreezenetworksadaptive,
      title={Grow-Prune-Freeze Networks: Adaptive & Continual Learning Technique for Olfactory Navigation}, 
      author={Kordel K. France and Ovidiu Daescu},
      year={2026},
      eprint={2605.25170},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.25170}, 
}
```
