import copy
import json
import math
import random

import torch

from utils.straggler_plan import fake_select, select_client
import numpy as np
from models.Update import LocalUpdate
from models.Aggreagtion import FedAvg_list, FedAvg_dict, FedAvgV1, FedAvg
from models.test import test_img

class Server(object):
    def __init__(self, args, global_net):
        self.args = args
        self.net = global_net

    def run(self):
        pass

    def train(self, dict_users, dataset_train):
        pass

    def test(self, dataset_test):
        pass