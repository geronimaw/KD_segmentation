from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersec=K.sum(y_true* y_pred)
    return (1-((2* intersec + 0.1) / (K.sum(y_true) + K.sum(y_pred) + 0.1)))

def iou(y_true,y_pred):
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    iou = (intersec + 0.1) / (union- intersec + 0.1)
    return iou