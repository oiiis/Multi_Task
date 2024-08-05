import torch
from torch import nn, optim
import copy

class Reptile:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, inner_steps=5):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.inner_steps = inner_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_update(self, support_set):
        support_inputs, support_targets = support_set
        for _ in range(self.inner_steps):
            self.model.train()
            support_outputs = self.model(support_inputs)
            support_loss = self.compute_loss(support_outputs, support_targets)

            self.model.zero_grad()
            support_loss.backward()
            for param in self.model.parameters():
                param.data -= self.lr_inner * param.grad.data

    def outer_update(self, original_params, updated_params):
        for param, updated_param in zip(self.model.parameters(), updated_params):
            param.data += self.lr_outer * (updated_param.data - param.data)

    def train_step(self, tasks):
        original_params = copy.deepcopy(list(self.model.parameters()))
        for support_set, _ in tasks:
            self.inner_update(support_set)

        updated_params = list(self.model.parameters())
        self.outer_update(original_params, updated_params)

    def compute_loss(self, outputs, targets):
        classification_loss = nn.CrossEntropyLoss()(outputs['classification'], targets['classification'])
        segmentation_loss = nn.BCEWithLogitsLoss()(outputs['segmentation'], targets['segmentation'])
        return classification_loss + segmentation_loss