import torch.nn.functional as F
import torch


def cross_entropy():
    pass

## for dice loss:
def compute_dice_iou(probs, targets_one_hot, epsilon=1e-6, return_iou=False):
    """
    Compute Dice score and optionally IoU score.

    Args:
        probs: Tensor of shape (N, C, H, W) - probabilities after softmax
        targets_one_hot: Tensor of shape (N, C, H, W) - one-hot encoded targets
        epsilon: smoothing factor
        return_iou: bool, whether to also compute and return IoU scores

    Returns:
        dice_score (scalar or per-class)
        optionally iou_score
    """
    dims = (0, 2, 3)  # sum over batch, height, width
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)

    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)
    
    # Handling binary vs multi-class Dice score averaging
    if probs.shape[1] == 2:
        dice_score = dice_per_class[1]# Only use foreground class
    else:
        dice_score = dice_per_class.mean()# Average over all classes

    if return_iou:
        union = cardinality - intersection
        iou_per_class = (intersection + epsilon) / (union + epsilon)
        if probs.shape[1] == 2:
            iou_score = iou_per_class[1]# Only foreground class
        else:
            iou_score = iou_per_class.mean()# Average over all classes
        return dice_score, iou_score
    
    return dice_score

def dice_loss(logits, targets, num_classes, epsilon=1e-6, return_metrics = False):
    """ 
    Computes multi-class Dice loss.

    Args:
        logits: Tensor of shape (N, C, H, W) — raw model outputs
        targets: Tensor of shape (N, H, W) — ground truth class indices
        num_classes: int — number of classes
        epsilon: float — smoothing factor to avoid division by zero

    Returns:
        Scalar Dice loss
    """
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float() # (N, C, H, W)
    # Apply softmax to logits
    probs = F.softmax(logits, dim=1)# (N, C, H, W)
    
    if return_metrics:
        # Calculate per-class Dice score using the helper function
        ### ?? Check whether this work as intended or not ??
        dice_score, iou_score = compute_dice_iou(probs, targets_one_hot, epsilon, return_iou=True)
        dice_loss_value = 1. - dice_score
        return dice_loss_value, dice_score, iou_score
    
    # otherwise: (this will not run if return_metrics=1)
    dice_score = compute_dice_iou(probs, targets_one_hot, epsilon, return_iou=False)
    dice_loss_value = 1. - dice_score
    return dice_loss_value


def log_cosh_dice_loss(logits, targets, num_classes, epsilon=1e-6, return_metrics=False):
    """
    Computes log-cosh of the multi-class Dice loss.

    Args:
        logits: Tensor of shape (N, C, H, W)
        targets: Tensor of shape (N, H, W)
        num_classes: int
        epsilon: float
        return_metrics: bool — whether to also return dice and IoU scores

    Returns:
        Scalar log-cosh Dice loss or tuple(loss, dice_score, iou_score)
    """
    if return_metrics:
        # dice_loss returns (loss, dice_score, iou_score)
        dice_loss_value, dice_score, iou_score = dice_loss(logits, targets, num_classes, epsilon, return_metrics=True)
        log_cosh_loss = torch.log(torch.cosh(dice_loss_value))
        return log_cosh_loss, dice_score, iou_score
    # Normal case
    dice_loss_value = dice_loss(logits, targets, num_classes, epsilon, return_metrics=False)
    log_cosh_loss = torch.log(torch.cosh(dice_loss_value))
    return log_cosh_loss
