import torch.nn.functional as F
import torch


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
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float().to(logits.device) # (N, C, H, W)

    # Apply softmax to logits
    probs = F.softmax(logits, dim=1) # (N, C, H, W)

    # Sum over batch, height, width
    dims = (0, 2, 3)    

    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    union = cardinality - intersection

    # Compute dice and iou for background, border, pet three classes
    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)
    iou_per_class = (intersection + epsilon) / (union + epsilon)

    # Handle imbalanced classes
    class_counts = torch.sum(targets_one_hot, dim=dims)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum()

    # Weight dice and iou
    weights = torch.tensor(class_weights, device=logits.device)
    weighted_dice = torch.sum(dice_per_class * weights) / weights.sum()
    weighted_iou = torch.sum(iou_per_class * weights) / weights.sum()

    if return_metrics:
        return 1.0 -  weighted_dice, weighted_dice, weighted_iou
    else:
        return 1.0 -  weighted_dice


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
