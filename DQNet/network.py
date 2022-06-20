# Includes Single Deep-Q Network and Double Deep-Q Network Structures
# Main Network File for Reinforcement Learning Application on Graph Coloring Problem

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from graph.graph_lib import Graph_Lib
from utils import freeze, unfreeze
from sac_model import SACActor, DoubleCritic
from buffer import ReplayBuffer

import copy
from torch.optim import Adam

def save_model(self, path):
        torch.save(self.state_dict(), path)
    
def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
class SACAgent():

    #network initialization
    def __init__(self, 
                 dimS, 
                 dimA, 
                 ctrl_range, 
                 gamma=0.99,
                 pi_lr=1e-4,
                 q_lr=1e-3,
                 polyak=1e-3,
                 alpha=0.2,
                 hidden1=400,
                 hidden2=300,
                 buffer_size=1000000,
                 batch_size=128,
                 device='cpu',
                 load_path=None):

        self.dimS = dimS
        self.dimA = dimA
        self.ctrl_range = ctrl_range

        self.gamma = gamma
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.polyak = polyak
        self.alpha = alpha

        self.batch_size = batch_size

        # pi: actor network, Q : 2 critic network
        self.pi = SACActor(dimS, dimA, hidden1, hidden2, ctrl_range).to(device)
        self.Q = DoubleCritic(dimS, dimA, hidden1, hidden2).to(device)

        # target network
        self.target_Q = copy.deepcopy(self.Q).to(device)

        freeze(self.target_Q)

        self.buffer = ReplayBuffer(dimS, dimA, limit=buffer_size)

        self.Q_optimizer = Adam(self.Q.parameters(), lr = self.q_lr)
        self.pi_optimizer = Adam(self.pi.parameters(), lr = self.pi_lr)

        self.device = device

        if(load_path != None): # for evaluation or further training, pretrained models can be loaded
            self.load_model(load_path)

    def load_model(self,path):
        print('networks loading...')
        checkpoint = torch.load(path)

        self.pi.load_state_dict(checkpoint['actor'])
        self.Q.load_state_dict(checkpoint['critic'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])
        self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return

    def save_model(self, path):
        print('adding checkpoints...')
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
                    {'actor': self.pi.state_dict(),
                     'critic': self.Q.state_dict(),
                     'target_critic': self.target_Q.state_dict(),
                     'actor_optimizer': self.pi_optimizer.state_dict(),
                     'critic_optimizer': self.Q_optimizer.state_dict()
                    },
                    checkpoint_path)

        return

    def evaluate(self, path): 
        # Real folder path is given, in where there are only graph files. 
        # All the graphs in the folder are taken as a single batch.
        graphs = Graph_Lib()
        node_cnts = graphs.read_batch(path)         # all graphs are read
        filenames = graphs.get_batch_filenames()    # filenames of all graphs in batch are taken
        batch = len(node_cnts) # number of graphs are taken as batch number
        # embedding initialization
        graphs.init_node_embeddings()
        graphs.init_graph_embeddings()  
        # parameter initialization
        colored_arrs = []   # for each graph in batch, colored information of each node is preserved
        max_node = -1       # maximum node count
        avg_node = 0        # average node count
        for cnt in node_cnts:
            avg_node += cnt
            if(cnt > max_node):
                max_node = cnt
            colored_arrs.append([False]*cnt)
        max_colors = [-1]*batch
        avg_node /= batch
        
        for colored_cnt in range(max_node): # until coloring all the nodes in all the graphs
            print("Number of Nodes Colored:", colored_cnt)
            # decides which nodes to color for each graph
            actions, _, _ = self.decide_node_coloring(graphs, node_cnts, batch, colored_arrs, colored_cnt, evaluate=True)
            # colors the selected nodes stored in actions
            colors = graphs.color_batch(actions)
            # sets max color for graphs to color in this step if the selected color is bigger than max color
            _ = self.get_rewards(batch, colors, max_colors)
            if(colored_cnt % 3 == 0): # updates graph embeddings at every 3 steps
                graphs.update_graph_embeddings()
        
        for f, c in zip(filenames, max_colors): # prints results
            print(f, ",", c+1)
        
        return max_colors
    
    def train(self, epochs, batch_size, min_nodes, max_nodes, path=None):
        # model is trained, if a path is set, then a batch from all the files in this path is set
        # otherwise random graphs are constructed within the node range 

        device = self.device

        for epoch in range(1, epochs + 1): # for each epoch

            loss_total = 0
            graphs = Graph_Lib()
            
            if(path != None): # if a path is set
                node_cnts = graphs.read_batch(path)
                batch = len(node_cnts) # number of graphs are taken as batch number
            else: # if no path is set
                batch = batch_size
                node_cnts = graphs.insert_batch(batch, min_nodes, max_nodes)
            
            # embedding initialization
            graphs.init_node_embeddings()
            graphs.init_graph_embeddings()
            
            # parameter initialization
            colored_arrs = []   # for each graph in batch, coloring informationf of each node is stored
            max_node = -1       # maximum nomber of nodes in the batch
            avg_node = 0        # average number of nodes in the batch
            for cnt in node_cnts:
                avg_node += cnt
                if(cnt > max_node):
                    max_node = cnt
                colored_arrs.append([False]*cnt)
            
            max_colors = [-1]*batch # max colors assigned for each graph in the batch are stored
            avg_node /= batch
            reward_last = np.zeros((3, batch)) # last 3 coloring steps' rewards are stored
            
            for colored_cnt in range(max_node): # until each node in each graph becomes colored
                # for each graph, nodes to color is seleced and stored in actions parameter
                actions, q_pred, not_finished = self.decide_node_coloring(graphs, node_cnts, batch, colored_arrs, colored_cnt) # not_finised = masks
                colors = graphs.color_batch(actions) # regarding to the selected actions (nodes), graphs are colored
                reward_last[colored_cnt % 3] = self.get_rewards(batch, colors, max_colors) # rewards for each graph are calculated for this step
                rewards = np.sum(reward_last, axis = 0) # last 3 steps' rewards are added and average of the mare taken
                rewards /= reward_last.shape[0]

                Q_loss, pi_loss = self.get_loss(graphs, batch, rewards, actions) # loss is calculated for the selected actions

                self.Q_optimizer.zero_grad()
                Q_loss.backward()
                self.Q_optimizer.step()

                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                self.pi_optimizer.step() 

                unfreeze(self.Q)

                self.step_counter += 1

                self.target_update()

                if(colored_cnt % 3 == 0): # at every 3 steps, graph embeddings are updated
                    graphs.update_graph_embeddings()
            
            #loss_total /= max_node # loss per node from whole coloring process is calculated
            #print("Epoch:", epoch, "--- Loss:", loss_total) 

            graphs.reset_batch()

            #if(epoch % 10 == 0): # set target as local
            #    self.target_qnet.load_state_dict(self.local_qnet.state_dict())  
            
            #if(epoch % 60 == 0):
            #    self.local_qnet.save_model('./backup/local_qnet' + str(epoch) + '.params')
            #    self.target_qnet.save_model('./backup/target_qnet' + str(epoch) + '.params')
                

            
    def decide_node_coloring(self, graphs, node_cnt, batch, colored_arrs, colored_cnt, evaluate=False):
        actions = []
        q_pred = []
        not_finished = batch
        for el in range(batch): # for each graph in batch:
            max_action = -9999
            max_node = -1
            # embeddings for nodes and graph is retrieved
            node_embeds = graphs.get_node_embed(el)
            graph_embeds = graphs.get_graph_embed(el)
            if(colored_cnt >= node_cnt[el]):
                # if a graph is finished coloring, then -1 is appended as action.
                # in C++, if action is seen as -1, no coloring will be made
                not_finished -= 1
                actions.append(-1)
                continue
            elif evaluate: # with the probability of 1-epsilon, network determines node to color
                node_np = np.array(node_embeds[:,6]) # incidence ordering (dynamic ordering) values for each node is taken
                node_np = node_np.argsort()[-10:][::-1] # nodes with maximum 10 incidence ordering values are chosen
                for node in node_np: # for each node selected above
                    if colored_arrs[el][node]:
                        continue
                    embeddings = np.concatenate([node_embeds[node], graph_embeds]) # final embedding is created
                    embeddings = torch.from_numpy(embeddings).float()
                    with torch.no_grad():
                        action = self.pi(embeddings, with_log_prob=False) # action value from the network for the current node is got

                    if(max_action < action): # node with the maximum action is saved
                        max_node = node
                        max_action = action

                colored_arrs[el][max_node] = True
                actions.append(max_node)
                q_pred.append(max_action)

            else: # with probability of epsilon, random uncolored node is selected for coloring
                found = False
                while not found:
                    node = random.randint(0, node_cnt[el] - 1) # random node is selected until an uncolored node is found
                    if not colored_arrs[el][node]:
                        found = True
                        colored_arrs[el][node] = True
                        embeddings = np.concatenate([node_embeds[node], graph_embeds]) # final embedding forthe selected node is created
                        embeddings = torch.from_numpy(embeddings).float()
                        with torch.no_grad():
                            action_val = self.pi(embeddings, with_log_prob=False) # action value for the node is calculated
                        q_pred.append(action_val)
                        actions.append(node)

        return actions, torch.Tensor(q_pred).requires_grad_(), not_finished
    
    def get_rewards(self, batch, colors, max_colors):
        # rewards are calculated for each graph in batch
        rewards = [0]*batch
        for el in range(batch):
            if(colors[el] == -1):
                # if a graph is completely colored already, then the color is returned from 
                # C++ function as -1. In this case, there will be no reward. Reward is set to an 
                # absurd value for eliminating in later steps
                rewards[el] = -9999
            else:
                # if maximum color number used in this step is increased for the selected graph, then
                # the increase amount is set as negative reward
                rewards[el] = - max(0, - max_colors[el] + colors[el])
                if(max_colors[el] < colors[el]):
                    # maximum color used for the selected graph is updated
                    max_colors[el] = colors[el]
        return np.array(rewards)

    def get_loss(self, graphs, batch, rewards, actions):
        losses = []
        for el in range(batch):
            
            # for each graph, embeddings are retrieved
            node_embeds = graphs.get_node_embed(el)
            graph_embeds = graphs.get_graph_embed(el)
            # total embedding is constructed
            embeddings = np.concatenate([node_embeds[actions[el]], graph_embeds])
            embeddings = torch.from_numpy(embeddings).float()
            #여기까진 state를 구하는 과정

            if(rewards[el] < -3):
                # if rewards is smaller than -3, that means that there is an absurd reward value,
                # therefore for the selected graph, no loss will be added
                continue
                
            with torch.no_grad():
                # loss is appended as reward + gamma * target network's prediction
                next_actions, log_probs = self.pi(next_embeddings, with_log_prob=True)

                target_q1, target_q2 = self.target_Q(next_embeddings, next_actions) #next_embedding prob
                target_q = torch.min(target_q1, target_q2)
                target = rewards[el] + self.gamma * (target_q - self.alpha * log_probs)

            out1, out2 = self.Q(embeddings, actions) # actions = act_batch? 

            Q_loss1 = torch.mean((out1 - target)**2)
            Q_loss2 = torch.mean((out2 - target)**2)
            Q_loss = Q_loss1 + Q_loss2

            losses.append(Q_loss) #embeddings 가 state다.
        return torch.Tensor(losses).requires_grad_()
