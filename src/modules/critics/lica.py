import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LICACritic(nn.Module):
    def __init__(self, scheme, args):
        super(LICACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = args.state_shape

        self.embed_dim = args.mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q = h2 + b2

        q = q.view(bs, -1, 1)

        return q


class LICACritic_CNN(nn.Module):
    def __init__(self, scheme, args):
        super(LICACritic_CNN, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = args.state_shape
        self.embed_dim = args.mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.mixing_embed_dim

        if self.state_dim == [6,20,20]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,4,stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16,32,3,stride=1),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(256,args.embed_dim),
                        nn.ReLU(),
                        )
        elif self.state_dim == [6,30,30]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(256,args.embed_dim),
                        nn.ReLU(),
                        )
        elif self.state_dim == [6,10,10]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=1),
                        nn.ReLU(),
                        nn.Conv2d(16,32,3,stride=1),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(64,args.embed_dim),
                        nn.ReLU(),
                        )
        else:
            raise Exception("Invalid Input Size for Convolutions: {}".format(self.state_dim))


        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(args.embed_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(args.embed_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(args.embed_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(args.embed_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(args.embed_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(args.embed_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim[0],self.state_dim[1],self.state_dim[2])
        state_embedding = self.embedding_network(states)

        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(state_embedding)
        b1 = self.hyper_b_1(state_embedding)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(state_embedding)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(state_embedding).view(-1, 1, 1)

        q = h2 + b2

        q = q.view(bs, -1, 1)

        return q
