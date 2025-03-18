import numpy as np

def compute_iou(pred_mask, gt_mask, num_classes):
    """
    Calcule l'IoU moyen et par classe entre un masque prédit et un masque ground truth.

    Args:
        pred_mask (np.array): Masque de segmentation prédit (H, W).
        gt_mask (np.array): Masque ground truth redimensionné (H, W).
        num_classes (int): Nombre de classes.

    Returns:
        dict: IoU moyen et IoU par classe.
    """
    iou_per_class = []

    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id).astype(np.uint8)
        gt_class = (gt_mask == class_id).astype(np.uint8)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union == 0:
            iou = None  # Classe absente dans les deux masques
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    # Filtrer uniquement les classes présentes dans au moins un masque
    valid_ious = [iou for iou in iou_per_class if iou is not None]
    
    if valid_ious:
        mean_iou = np.mean(valid_ious)
    else:
        mean_iou = None  # Aucune classe pertinente pour le calcul de l'IoU

    return {"mean_iou": mean_iou, "iou_per_class": iou_per_class}
