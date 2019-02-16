from keras import backend as K
import tensorflow as tf

def triplet_loss(y_true, y_pred, cosine = True, alpha = 0.2, embedding_size = 128):
    ind = int(embedding_size * 2)
    a_pred = y_pred[:, :embedding_size]
    p_pred = y_pred[:, embedding_size:ind]
    n_pred = y_pred[:, ind:]
    if cosine:
        positive_distance = 1 - K.sum((a_pred * p_pred), axis=-1)
        negative_distance = 1 - K.sum((a_pred * n_pred), axis=-1)
    else:
        positive_distance = K.sqrt(K.sum(K.square(a_pred - p_pred), axis=-1))
        negative_distance = K.sqrt(K.sum(K.square(a_pred - n_pred), axis=-1))
    loss = K.maximum(0.0, positive_distance - negative_distance + alpha)
    return loss

def attribute_crossentropy(y_true, y_pred):
    y_true = tf.gather(labels, tf.to_int32(y_true[0]))
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)