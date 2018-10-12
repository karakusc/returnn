import tensorflow as tf

class BMUFOptimizer(tf.train.Optimizer):
  def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
               device_sparse='', bmuf_every=8, block_momentum=0.9, block_lr=1.0):
    if name is None:
      name = "BMUF{}".format(type(optimizer).__name__)
    self._opt = optimizer
    self._device_dense = device_dense
    self._device_sparse = device_sparse
    self._bmuf_every = bmuf_every
    self._block_momentum = block_momentum
    self._block_lr = block_lr
    super(BMUFOptimizer, self).__init__(name=name, use_locking=use_locking)

  def apply_gradients(self, grads_and_vars, global_step=None, name='apply_gradients'):
    must_apply_bmuf = tf.equal(tf.mod(global_step, self._bmuf_every), tf.constant(0, dtype=tf.int64))
    in_block_op = self._opt.apply_gradients(grads_and_vars, global_step, name)
    block_end_op = _apply_gradients_block_end(grads_and_vars, global_step, name)
    op = tf.cond(must_apply_bmuf, true_fn=block_end_op, false_fn=in_block_op)
    return op

  def compute_gradients(self, loss, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
          colocate_gradients_with_ops=False, grad_loss=None):
      return self._opt.compute_gradients(loss, var_list, gate_gradients, aggregation_method,
              colocate_gradients_with_ops, grad_loss)

  def _apply_gradients_block_end(self, grads_and_vars, global_step, name):
    apply_grad = self._opt.apply_gradients(grads_and_vars, global_step, name)
    assigns = []
    with tf.control_dependencies([apply_grad]):
      from horovod.tensorflow import allreduce, size
      if size() > 1:
        with tf.name_scope("bmuf"):
          for grad, var in grads_and_vars:
            if grad is not None:
              var_avg = allreduce(var, global_op=True)

              prev_weight = self._opt._get_or_make_slot(var, var.initialized_value(), 'prev_weight', name)
              prev_weight_g = self._opt._get_or_make_slot(var, var.initialized_value(), 'prev_weight_g', name)
              prev_delta = self._opt._zeros_slot(var, "prev_delta", self._name)

              G_t = var_avg - prev_weight_g

              curr_delta = self._block_momentum * prev_delta + self._block_lr * G_t
              assigns.append(prev_delta.assign(curr_delta))

              var_avg = prev_weight + curr_delta 
              assigns.append(prev_weight.assign(var_avg))

              var_avg_g = var_avg + self._block_momentum * curr_delta
              assigns.append(prev_weight_g.assign(var_avg_g))
              assigns.append(var.assign(var_avg_g))

    return tf.group(*assigns)

