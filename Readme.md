# Project 1: Navigation


### Background

For this project, the Unity ML-Agents Banana simulated environment was used to train an agent to navigate (and collect bananas) 
in a large, square world. A reward of +1 is awarded for collecting a yellow banana while -1 is awarded for 
collecting a blue banana.


### Environment Details
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around 
agent's forward direction.

The agent will take one of four discrete actions at each time step i.e. walk forward or backward and turn left or right. 

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The environment is considered solved if the agent is able to receive an average reward (over 100 consecutive episodes) of at least +13.


### System Setup on Linux Machine
Step 1: Clone the DRLND Repository [click here] (https://github.com/udacity/deep-reinforcement-learning#dependencies)
Step 2: Download the Unity Environment [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Step 3: Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.
Step 4: Clone the GitHub repository for this project into the `p1_navigation/` folder.
Step 5: Download and install Anaconda [click here] (https://www.anaconda.com/distribution/)
Step 6: Create and activate a virtual environment with the following libraries:
* UnityAgents (ver. 0.4.0) [click here] (https://pypi.org/project/unityagents/) for more information
* Pandas [click_here] (https://anaconda.org/anaconda/pandas) for more information
* Numpy [click here] (https://anaconda.org/anaconda/numpy) for more information
* Random [click here] (https://pypi.org/project/random2/) for more information
* Sys 
* Torch [click here] (https://pytorch.org/) for more information
* Matplotlib [click here] (https://anaconda.org/conda-forge/matplotlib) for more information
* Collections [click here] (https://anaconda.org/lightsource2-tag/collection) for more information


### Instructions
Step 1: Start Jupyter Notebook
Step 2: Navigate to the folder for this project (in the `p1_navigation/` folder) 
Step 3: Open any of the jupyter notebooks
Step 4: Change Kernel to 'drlnd'
Step 5: Run the cells as required 


### Notebook Details:

1) Deep Q-Learning: Run Navigation.ipynb
2) Double Deep Q-Learning: Run Navigation_double_dqns.ipynb
3) Double Deep Q-Learning with Dueling: Run Navigation_double_dueling_dqns.ipynb
4) Double Deep Q-Learning with Dueling and Prioritized Experience Replay (PER): Run Navigation_double_dueling_PER_dqns.ipynb


### Other Information:

Refer to Report.pdf for more information on the learning algorithms, hyperparameters, architecture for neural network models etc.