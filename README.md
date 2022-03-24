# Simulation of individual replay strategies with an autonomously learned state decomposition
To study the implications of offline learning in spatial navigation, from rodents' behavior to robotics, we have
investigated the role of two Model Free (MF) - Reinforement Learning (RL) replay techniques in a circular maze,
consistent with the original Morris water maze task (Morris, 1981) in terms of environment/robot size ratio. The
learning performances of the analyzed replay techniques are tested here in two main conditions:
- A deterministic version of the task, where an action *a* performed in a state *s* will always lead the robot to the
  same arrival state *s'* with probability 1.
- A stochastic version of the task, where performing action *a* in state *s* is associated to non-null probabilities of
  arrival for more than one state.
  
> This code goes with the following submission: Massi et al. (2022) Model-based and model-free
> replay mechanisms for reinforcement learning in neurorobotics. Submitted.

## Usage
- `main.py` is used to run the whole simulation: analysis on the learning phase and evaluation over different values of
  learning rate *alpha*. There you can set all the parameters of your simulation and choose if you want to simulate the
  deterministic or stochastic environment.
- `functions.py` contains all the functions used in `main.py`.
- `read_json_alpha.py` finds the best value of the learning rate *alpha* from the .json file generated from `main.py`.
- `learning_performances_alpha_selection_figures.py` produces the plot concerning the learning phase for all the types
  of replay and for both the deterministic and stochastic environment, starting from the .json file generate from 
  main.py`.
- `analyse_statistic.py` performs an analysis of the statistical significance of the learning perfomances of the tested
  algorithms, saves these results in a .json file and plots them in a way that they can be added to the first figure
  obtained in `learning_performances_alpha_selection_figures.py`.
- `qvalue_map_subplots.py` produces the plot concerning the propagation of the maximal Q-values in the maze (for
  individual 50 and trial 3), for all the types of replay and for both the deterministic and stochastic environment (to
  be selected at the beginning of the file). The .json file generated from `main.py` is needed.
- `histogram_qvalues_propagation.py` produces the histograms plots and the statistical analysis concerning the
  propagation of the maximal Q-values in the maze (for instance as a function of the distance to the rewarding state,
  for individual 50 at trial 3, for all the indivuals at trial 3 and divided in bins), for all the types
  of replay and for both the deterministic and the stochastic environment. The .json files generated from `main.py` are
  needed.
- `entropy_map.py` produces a figure representing the maximal entropy for each state of the environment, in the
  stochastic case. The transitions are read from `transitions.txt`, collected after a long random exploration in ROS
  Gazebo. The range to which the entropy values are normalized is the same used for visualizing the entropy map of the
  other experiment in the paper (Sect. 4).
- the **data_files** folder contains the all the files generated from the ROS Gazebo experiments which are needed to run
the simulations here, and it is also the destination folder when the .json files with the results are saved.
  
## Questions?
Contact Elisa Massi (lastname (at) isir (dot) upmc (dot) fr)