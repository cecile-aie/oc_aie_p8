import tensorflow as tf
from tensorflow.keras import backend as K

##### METRIQUE #################################################

def iou_mean(y_true, y_pred, smooth=1e-6):
    """
    Calcul de l'IoU pixel-wise pour des masques one-hot (y_true)
    et des probabilités softmax (y_pred)
    Moyenne de IoU sur toutes les classes
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)  # Moyenne sur le batch

##### METRIQUE #################################################

def tversky_loss(alpha=0.7, beta=0.3):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Éviter division par zéro
        
        # Vérifier si y_true est one-hot, sinon le convertir
        if len(y_true.shape) == 3:  # (batch_size, height, width)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]) 

        # Calculer TP, FP et FN pixel par pixel
        TP = y_true * y_pred
        FP = (1 - y_true) * y_pred
        FN = y_true * (1 - y_pred)
        
        # Réduction sur les axes de l'image et des classes, mais pas sur le batch
        TP = tf.reduce_sum(TP, axis=-1)  # Garder (batch_size, height, width)
        FP = tf.reduce_sum(FP, axis=-1)
        FN = tf.reduce_sum(FN, axis=-1)
        
        # Calculer l'indice de Tversky par pixel
        tversky_index = TP / (TP + alpha * FP + beta * FN + 1e-7)

        # Retourner une perte compatible avec sample_weights (batch, height, width)
        return 1 - tversky_index
    
    return loss
    

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for multilabel classification.
    Parameters:
    gamma -- focusing parameter. Default is 2.
    alpha -- balancing parameter. Default is 0.25, can be a class_weights np array
    """
    def focal_loss_fixed(y_true, y_pred):
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred + K.epsilon())
        
        # Calculate the focal weight
        weight = alpha * K.pow(1 - y_pred, gamma)
        
        # Apply the weight to cross entropy
        focal_loss = K.sum(weight * cross_entropy, axis=-1)
        return focal_loss
    return focal_loss_fixed