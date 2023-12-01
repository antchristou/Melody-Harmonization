import torch.nn.functional as F
#import evaluation_helpers
import torch

class Trainer:

    def __init__(self, model,
                 optimizer, 
                 loss_fn,
                 train_dataloader,
                 test_dataloader,
                 device,
                 train_losses=[],
                 test_losses=[]):
 
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn  
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.train_losses = train_losses
        self.test_losses = test_losses
    

    def run_epoch(self):

        for inputs,targets in self.train_dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # forward pass

            # create mask for target sequence
            target_input = targets[:,:-1]
            target_expected = targets[:,1:]

            tgt_mask = self.model.get_tgt_mask(target_input.size(1)).to(self.device)

            output = self.model(inputs,target_input, tgt_mask)
            output = output.permute(0,2,1) 

            loss = self.loss_fn(output, target_expected)

            # compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # lr_scheduler.step()

        print("Loss:",loss.item())
        self.train_losses.append(loss.item())

    def run_test_epoch(self):

        for inputs,targets in self.test_dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # forward pass
            # create mask for target sequence

            target_input = targets[:,:-1]
            target_expected = targets[:,1:]

            tgt_mask = self.model.get_tgt_mask(target_input.size(1)).to(self.device)

            output = self.model(inputs,target_input, tgt_mask)
            output = output.permute(0,2,1)

            loss = self.loss_fn(output, target_expected)


        print("Validation Loss:",loss.item())
        self.test_losses.append(loss.item())

    def train(self,num_epochs):
        for i in range(num_epochs):
            print("Epoch",str(i)+":")
            self.run_epoch()
            if i % 5 == 0:
                self.run_test_epoch()
