# maml.py

import torch
from torch import nn, optim

class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_update(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        updated_params = [param - self.lr_inner * grad for param, grad in zip(self.model.parameters(), grads)]
        return updated_params

    def outer_update(self, task_losses):
        meta_loss = torch.stack(task_losses).mean()
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

    def train_step(self, tasks):
        task_losses = []
        for support_set, query_set in tasks:
            support_inputs, support_targets = support_set
            query_inputs, query_targets = query_set

            # Compute loss on support set
            self.model.train()
            support_outputs = self.model(support_inputs)
            support_loss = self.compute_loss(support_outputs, support_targets)

            # Compute updated parameters
            updated_params = self.inner_update(support_loss)

            # Compute loss on query set with updated parameters
            query_outputs = self.model(query_inputs)
            query_loss = self.compute_loss(query_outputs, query_targets)

            task_losses.append(query_loss)

        self.outer_update(task_losses)

    def compute_loss(self, outputs, targets):
        # Ensure targets['segmentation'] is the same size as outputs[0]
        if targets['segmentation'].size() != outputs[0].size():
            targets['segmentation'] = nn.functional.interpolate(targets['segmentation'].unsqueeze(1).float(), size=outputs[0].shape[2:], mode='trilinear', align_corners=False)

        classification_loss = nn.CrossEntropyLoss()(outputs[1], targets['classification'].long())
        segmentation_loss = nn.BCEWithLogitsLoss()(outputs[0], targets['segmentation'])
        return classification_loss + segmentation_loss
