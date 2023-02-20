# EvoESN: source code and daat files for running experiments of the EvoESN model.

This repository contains the necessary source code to use the EvoESN with the PSO algorithm. Also there are files with data, for making easy experiments. 
For performing an experiment the user can run in a terminal the command: 
"python main.py". 
By default, it will solve the problem related to Lorenz system. It is possible to select the problem directly in the main.py file. 
The global parameters of the algorithm and the parameters of the model are set in three main structures: ESN_param, Learning_config and PSO_param. The user can modify the parameters directly changing the values on those data structures.
Note that work over this code should cite our first article regarding this model: <br />

@INPROCEEDINGS{BasterIJCNN2022, <br />
  author={Basterrech, Sebastian and Rubino, Gerardo}, <br />
  booktitle={2022 IEEE International Joint Conference on Neural Networks (IJCNN)}, <br />
  title={{Evolutionary Echo State Network: evolving reservoirs in the Fourier space}}, <br />
  year={2022}, <br />
  volume={}, <br />
  number={}, <br />
  pages={1-8}, <br />
  doi={10.1109/IJCNN55064.2022.9892892}} <br />
  

## Notes<br/>
If you find any problems or you have a suggestion for improvement, please don't hesitate in writing me. It will help us make this code better for everybody.
