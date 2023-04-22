import tensorflow as tf
from tensorflow.keras import backend

def TripletLossAccuracy(top_n=1):
    
    # y_true: (1D array) - [1, 1, 3, 4, 5, 5]
    # y_pred: (2D array) - embeddings of each image
    # returns the accuracy of the predictions within the batch (i.e., is 1st image most similar to the 2nd one?)
    # note - in case of no positive example as for id=3 and id=4 then we don't add it as bad record
    
    def triplet_loss_accuracy(y_true, y_pred):
        batch_size = tf.size(y_true)
        pdist_matrix = backend.sqrt(backend.sum((y_pred - y_pred[:, None]) ** 2, axis=-1))
        adjacency = tf.cast(backend.equal(y_true, y_true[:, None]), dtype=tf.float32)
        adjacency -= tf.eye(batch_size)
        total_positive = backend.sum(tf.cast(backend.any(adjacency, axis=-1), dtype=tf.float32))
        top_n_matches = tf.argsort(tf.where(pdist_matrix > 0, pdist_matrix, tf.ones_like(pdist_matrix) * float('inf')))[:, :top_n]
        y = tf.range(batch_size)
        tiled_y = tf.tile(y[:, None], [1, top_n])
        indices = tf.reshape(tf.transpose(tf.stack([tiled_y, top_n_matches])), (-1, 2))
        tensor_best_n = tf.zeros(shape=(batch_size, batch_size), dtype=tf.float32)
        tensor_best_n = tf.tensor_scatter_nd_update(tensor_best_n, indices, tf.ones(len(indices), dtype=tf.float32))
        correct_predictions = backend.sum(tf.cast(backend.sum(adjacency * tensor_best_n, axis=-1) > 0, dtype=tf.float32))
        return correct_predictions / (total_positive + 1e-6)
    
    return triplet_loss_accuracy
