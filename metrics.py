import tensorflow as tf
from tensorflow.keras import backend

def TripletLossAccuracy(top_n=1):
    
    # y_true: (1D array) - [1, 1, 3, 4, 5, 5]
    # y_pred: (2D array) - embeddings of each image
    # returns the accuracy of the predictions within the batch (i.e., is 1st image most similar to the 2nd one?)
    # note - in case of no positive example as for id=3 and id=4 then we don't add it as bad record
    
    def triplet_loss_accuracy(y_true, y_pred):
        lshape = tf.shape(y_true)
        batch_size = lshape[0]
        y_true = tf.reshape(y_true, [batch_size, 1])
        adjacency = tf.math.equal(y_true, tf.transpose(y_true))
        adjacency = tf.cast(adjacency, dtype=tf.float32)

        pdist_matrix = backend.sqrt(backend.sum((y_pred - y_pred[:, None]) ** 2, axis=-1))
        total_positive = tf.math.count_nonzero(K.sum(adjacency, axis=-1) - 1, dtype=tf.float32)
        top_n_matches = tf.argsort(tf.where(pdist_matrix > 0, pdist_matrix, tf.ones_like(pdist_matrix) * float('inf')))[:, :top_n]
        y = tf.range(batch_size)
        tiled_y = tf.tile(y[:, None], [1, top_n])
        indices = tf.reshape(tf.transpose(tf.stack([tiled_y, top_n_matches])), (-1, 2))
        tensor_best_n = tf.zeros(shape=(batch_size, batch_size), dtype=tf.float32)
        tensor_best_n = tf.tensor_scatter_nd_update(tensor_best_n, indices, tf.ones(len(indices), dtype=tf.float32))

        correct_predictions = tf.math.count_nonzero(backend.sum(adjacency * tensor_best_n, axis=-1), dtype=tf.float32)
        return correct_predictions / (total_positive + backend.epsilon())
    
    return triplet_loss_accuracy
