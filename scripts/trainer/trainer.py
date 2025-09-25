# Define a general trainer and split class for the models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os  # likely for model saving/loading
from tqdm import tqdm
from typing import Optional
from pathlib import Path

class trainer:
    def __init__(self, model, 
                optimizer, 
                criterion, num_epoch,
                dataloaders, 
                model_path,
                device,
                use_trap_scheduler=False,
                model_kwargs = None,
                criterion_kwargs= {"num_classes": 2, "epsilon": 1e-6},
                backbone_frozen: bool = False,
                freeze_epochs: Optional[int] = None):
        """
    Initializes the trainer object with all necessary components for training and validation.

    Args:
        model: The PyTorch model to be trained.
        optimizer: The optimizer used to update model parameters.
        criterion: Loss function used for both training and validation.
        num_epoch: Number of epochs to train the model.
        dataloaders (dict): Dictionary containing 'train' and split DataLoader objects.
        use_trap_scheduler (bool): Whether to use a trapezoidal learning rate scheduler.
        device (str): Device to use for training ('cuda' or 'cpu') (default: 'cuda').
        model_kwargs (dict, optional): Additional keyword arguments to pass to the model during the forward pass.
        criterion_kwargs (dict, optional): Additional keyword arguments to pass to the criterion/loss function. 
                                        Defaults to {"num_classes": 2, "epsilon": 1e-6}.

    Initializes:
        - Tracking lists for training and validation loss.
        - Tracking lists for validation Dice and IoU metrics.
        - Dataset sizes for weighted loss/metric computation.
    """
        
        self.model = model
        self.optimizer = optimizer
        # learning rates of 2 different groups:
        self.lr_groups = [pg['lr'] for pg in optimizer.param_groups]# self.lr_groups[0] is LR for backbone and self.lr_groups[1] is LR for head
        self.criterion = criterion
        self.criterion_kwargs= criterion_kwargs 
        self.num_epoch = num_epoch
        self.dataloaders = dataloaders
        self.device = device
        self.model_path = model_path
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        
        # Calculate and store dataset sizes:
        self.dataset_sizes = {split: len(dataloader.dataset) for split, dataloader in dataloaders.items()}

        # Following is needed for trapezoidal scheduler 
        # these are for trapezoidal scheduler
        if use_trap_scheduler:
            self.num_batches_per_epoch = len(self.dataloaders['train'])
            self.total_batch_updates = self.num_batches_per_epoch * self.num_epoch
            self.current_batch_step = 0
            self._setup_trapezoid_scheduler()

        # These are for the case when one wants to freeze the vit-backbone for a few epochs initially
        self.backbone_frozen = backbone_frozen
        if self.backbone_frozen:
            if freeze_epochs is None:
                raise ValueError("freeze_epochs must be provided if backbone_frozen=True")
            self.freeze_epochs = freeze_epochs
            # freeze backbone initially
            self.freeze_backbone() #DEFINE THESE FUNCTIONS
            self.reinit_optimizer(head_only=True)#DEFINE THESE FUNCTIONS
        else:
            self.freeze_epochs = None
            # train all parameters from the start
            # self.reinit_optimizer(head_only=False) # I already have self.optimizer initialised above, so no need to add it here
            
        ## error terms:
        self.train_error_epoch_list = []
        self.val_error_epoch_list = []

        ## metrics lists:
        self.val_dice_epoch_list = []
        self.val_iou_epoch_list = []

    ## 
    def _setup_trapezoid_scheduler(self):
        self.warmup_steps = int(0.05 * self.total_batch_updates)
        self.decay_steps = int(0.2 * self.total_batch_updates)
        self.plateau_steps = self.total_batch_updates - self.warmup_steps - self.decay_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._trapezoid_lr_lambda)# it adjusts the learning rate according to the trapezoidal schedule instead of using fixed steps like StepLR or ExponentialLR.

    ## trapezoidal scheduler as a private method
    def _trapezoid_lr_lambda(self, batch_step):
        if batch_step < self.warmup_steps:
            return float(batch_step) / max(1, self.warmup_steps)
        elif batch_step < (self.warmup_steps + self.plateau_steps):
            return 1.0
        else:
            return max(0.0, float(self.total_batch_updates - batch_step) / max(1, self.decay_steps))

    ## --- NEXT 3 FUNCS are to help with initial freezing and later unfreezing of the vit backbone
    def freeze_backbone(self):
        for p in self.model.backbone_parameters:
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.model.backbone_parameters:
            p.requires_grad = True
    
    def reinit_optimizer(self, head_only=False):
        OptimClass = type(self.optimizer)
        if head_only:
            params = self.model.head_parameters
            if len(self.lr_groups)==1:
                # when the input parameter of optimizer is mentioned as vit_seg_model.parameters()
                lr = self.lr_groups[0]
                self.optimizer = OptimClass([{"params": params, "lr": lr}])
            elif len(self.lr_groups)==2: 
                # different learning rate for training of backbone and head (backbone and head parameters are seperate input for the optimizer obviously). 
                lr = self.lr_groups[1]  # head LR
                self.optimizer = OptimClass([{"params": params, "lr": lr}])
        else:
            if len(self.lr_groups)==1:
                # when the input parameter of optimizer is mentioned as vit_seg_model.parameters()
                params = [
                    {"params": self.model.parameters(), "lr": self.lr_groups[0]}
                        ]
            elif len(self.lr_groups)==2:
                # different learning rate for training of backbone and head (backbone and head parameters are seperate input for the optimizer obviously).
                params = [
                    {"params": self.model.backbone_parameters, "lr": self.lr_groups[0]},
                    {"params": self.model.head_parameters, "lr": self.lr_groups[1]}
                        ]
            self.optimizer = OptimClass(params)


    ## these are the train, train_epoch and val_epoch functions we need:
    def train(self, k=1):
        for epoch in tqdm(range(self.num_epoch), desc="Epochs"):
            ## unfreeze after freeze_epochs
            if self.backbone_frozen and epoch == self.freeze_epochs:
                print(f" Unfreezing backbone at epoch {epoch+1}")
                self.unfreeze_backbone()
                self.reinit_optimizer(head_only=False)

            # train and validation step for i'th epoch
            avg_epoch_train_loss = self.train_epoch(epoch=epoch)
            avg_epoch_val_loss, avg_epoch_val_dice, avg_epoch_val_iou = self.val_epoch()# unpack all three values returned by val_epoch
            
            # accumulate losses
            self.train_error_epoch_list.append(avg_epoch_train_loss)
            self.val_error_epoch_list.append(avg_epoch_val_loss)

            # accumulate metrics
            self.val_dice_epoch_list.append(avg_epoch_val_dice)
            self.val_iou_epoch_list.append(avg_epoch_val_iou)

            if (epoch + 1) % k == 0:
                print(f"Epoch [{epoch+1}/{self.num_epoch}] - Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {avg_epoch_val_loss:.4f} | Dice score: {avg_epoch_val_dice} |IOU score: {avg_epoch_val_iou:.7f} ")


    def train_epoch(self, epoch=None):
        # training mode
        self.model.train()  
        # initialise cumulative loss
        loss_ith_epoch_minibatch_cummul = 0.0
        
        # for minibatch_input, truth in self.dataloaders['train']:
        for minibatch_input, truth in tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1} - Training", leave=False):
            # move data to device
            minibatch_input, truth = minibatch_input.to(self.device), truth.to(self.device)
            # zero gradients
            self.optimizer.zero_grad()
            # forward pass
            output_minibatch = self.model(minibatch_input, **self.model_kwargs)
            # calculate loss
            loss_ith_epoch_minibatch = self.criterion(output_minibatch, truth, **self.criterion_kwargs)
            # backward pass
            loss_ith_epoch_minibatch.backward()
            # optimizer step
            self.optimizer.step()
            # update learning rate if using trapezoidal scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step(self.current_batch_step)
                self.current_batch_step += 1
            # accumulate the loss for average calculation later
            batch_size = minibatch_input.size(0)
            loss_ith_epoch_minibatch_cummul += loss_ith_epoch_minibatch.item() * batch_size

        # calculate average train loss for the epoch
        avg_epoch_train_loss = loss_ith_epoch_minibatch_cummul / self.dataset_sizes['train'] if self.dataset_sizes['train'] > 0 else 0 
        # return weighted average train epoch loss
        return avg_epoch_train_loss

    def val_epoch(self, split='val'):
        # evaluation mode
        self.model.eval()  
        # initialise cumulative loss and num_minibatch counter
        loss_ith_epoch_minibatch_val_cummul = 0.0
        dice_cum = 0.0 # cumulative
        iou_cum = 0.0 # cumulative

        # no-grad mode
        with torch.no_grad():
            for minibatch_input, truth in self.dataloaders[split]:
                # move to device
                minibatch_input, truth = minibatch_input.to(self.device), truth.to(self.device)
                # forward pass
                output_val_minibatch = self.model(minibatch_input, **self.model_kwargs)

                # validation loss
                loss_val, dice_score, iou_score = self.criterion(output_val_minibatch, truth,
                                    **{**self.criterion_kwargs, "return_metrics": True})
                
                batch_size = minibatch_input.size(0)
                # accumulate loss for average caln later
                loss_ith_epoch_minibatch_val_cummul += loss_val.item() * batch_size
                # accumulate metrics
                dice_cum += (dice_score.item() if isinstance(dice_score, torch.Tensor) else dice_score) * batch_size
                iou_cum += (iou_score.item() if isinstance(iou_score, torch.Tensor) else iou_score) * batch_size

        # calculate average val loss for the epoch
        avg_epoch_val_loss = loss_ith_epoch_minibatch_val_cummul / self.dataset_sizes[split] if self.dataset_sizes[split] > 0 else 0
        # calculate average metrics
        avg_dice = dice_cum / self.dataset_sizes[split] if self.dataset_sizes[split] > 0 else 0
        avg_iou = iou_cum / self.dataset_sizes[split] if self.dataset_sizes[split] > 0 else 0

        return avg_epoch_val_loss, avg_dice, avg_iou # return average val epoch loss
        #return avg_epoch_val_loss 

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Save model to path {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Load model from {self.model_path}")