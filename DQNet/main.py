# import networkx as nx
import sys
from network import *

if __name__ == '__main__':
    
    # PARAMETER INITIALIZATION
    learning_rate = 0.005
    batch_size = 3
    input_size = 24
    action_size = 1
    epochs = 120
    min_nodes = 100
    max_nodes = 500

    # TRAINING 
    dqn = SACAgent(input_size, action_size,max_nodes)
    dqn.train(epochs, batch_size, min_nodes, max_nodes)
    
    # EVALUATION
    backup_loc = './backup/' # backup location where the model saved will be loaded
    eval_path = '../Matrices/small/' # folder path matrices for evaluation (will be taken as single batch)
    dqn = SACAgent(input_size, learning_rate, max_nodes, load_path = backup_loc)
    dqn.evaluate(eval_path)