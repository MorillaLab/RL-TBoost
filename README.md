# RL-TBoost
Reinforcement Learning Enhanced by Topological Data Analysis (TDA) to predict Y1 risk of mortality after lung transplantation.


In Reinforcement Learning (RL), a state serves as an observable entity conveying valuable information to the agent, which employs this information to make optimal decisions based on its policy. States can be of finite or infinite nature.

To facilitate our research, we have constructed a custom virtual environment using TensorFlow. In this environment, the state plays a pivotal role in assessing the model's accuracy, which is built upon a deep learning framework. This deep learning model is intricately linked to invariant multiscale features derived from topological data analysis (TDA) of our dataset. Consequently, during both the training and testing phases of the model, the dataset's shape is taken into consideration as we endeavor to distinguish between different classes.

For instance, suppose we aim to train an agent for this model:

During each turn, the agent has two possible actions: running the model or terminating the current round. The overarching objective is to ensure that the previous loss surpasses the new loss concerning the training set. This condition signifies the preservation of the dataset's shape. A representative environment for implementing this scenario is as follows:

Actions: We define two actions, denoted as Action 0 (running the model) and Action 1 (terminating the current round). 

Observations: The observation comprises the loss of the testing dataset at epoch 'n,' which should be less than the loss at epoch 'n-1.' 

Reward: Our primary goal is to maximize the accuracy of the test dataset.
