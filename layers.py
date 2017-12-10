import tensorflow as tf

wd = 5e-4
data_dict = np.load("./data/pretrained/vgg16.npy", encoding='latin1').item()

def batch_norm_layer(x, train_phase, scope_bn):
    return tensorflow.contrib.layers.python.layers.batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def _max_pool( bottom, name, debug):
    pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

    if debug:
        pool = tf.Print(pool, [tf.shape(pool)],
                        message='Shape of %s' % name,
                        summarize=4, first_n=1)
    return pool

def _conv_layer( bottom, train_phase, name, batch_norm=False):
    with tf.variable_scope(name) as scope:
        filt = get_conv_filter(name)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        if batch_norm:
            conv = batch_norm_layer(conv, train_phase, 'bn-'+name)
        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        # Add summary to Tensorboard
        _activation_summary(relu)
        return relu

def _fc_layer( bottom, name, num_classes=None,
              relu=True, debug=False,use=""):
    with tf.variable_scope(name) as scope:
        
        if use=="vgg":
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            bottom = tf.reshape(bottom, [-1, dim])
            if name == 'fc6':
                filt = get_fc_weight(name)
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = get_fc_weight_reshape(name, [4096, 1000],num_classes=num_classes)
            else:
                filt = get_fc_weight(name)
            conv = tf.matmul(bottom, filt)
        else:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            _add_wd_and_summary(filt, wd, "fc_wlosses")
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, num_classes=num_classes)
        bias = tf.nn.bias_add(conv, conv_biases)

        if relu:
            bias = tf.nn.relu(bias)
        _activation_summary(bias)

        if debug:
            bias = tf.Print(bias, [tf.shape(bias)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return bias

def _score_layer( bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        #print name,bottom.get_shape().as_list()
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input)**0.5
        #elif name == "score_pool4":
        #    stddev = 0.001
        #elif name == "score_pool3":
        #    stddev = 0.0001
        else:
            stddev = 0.001
        # Apply convolution
        w_decay = wd

        weights = _variable_with_weight_decay(shape, stddev, w_decay,
                                                   decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias

def _upscore_layer( bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = get_deconv_filter(f_shape)
        _add_wd_and_summary(weights, wd, "fc_wlosses")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    _activation_summary(deconv)
    return deconv

def get_deconv_filter( f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def get_conv_filter( name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % str(shape))
    var = tf.get_variable(name="filter", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:

        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,name='weight_loss')
        #weight_decay = tf.mul(tf.nn.l2_loss(var), wd,name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def get_bias( name, num_classes=None):
    bias_wights = data_dict[name][1]
    shape = data_dict[name][1].shape
    if name == 'fc8':
        bias_wights = _bias_reshape(bias_wights, shape[0],
                                         num_classes)
        shape = [num_classes]
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="biases", initializer=init, shape=shape)
    _variable_summaries(var)
    return var

def get_fc_weight( name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def _bias_reshape( bweight, num_orig, num_new):
    """ Build bias weights for filter produces with `_summary_reshape`
    """
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight

def _summary_reshape( fweight, shape, num_new):
    """ Produce weights for a reduced fully-connected layer.
    FC8 of VGG produces 1000 classes. Most semantic segmentation
    task require much less classes. This reshapes the original weights
    to be used in a fully-convolutional layer which produces num_new
    classes. To archive this the average (mean) of n adjanced classes is
    taken.
    Consider reordering fweight, to perserve semantic meaning of the
    weights.
    Args:
      fweight: original weights
      shape: shape of the desired fully-convolutional layer
      num_new: number of new classes
    Returns:
      Filter weights for `num_new` classes.
    """
    lastdim=len(shape)-1
    num_orig = shape[lastdim]
    shape[lastdim] = num_new
    assert(num_new < num_orig)
    n_averaged_elements = num_orig//num_new
    avg_fweight = np.zeros(shape)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        if lastdim==3:
            avg_fweight[:, :, :, avg_idx] = np.mean(fweight[:, :, :, start_idx:end_idx], axis=3)
        if lastdim==1:
            avg_fweight[:, avg_idx] = np.mean(fweight[:, start_idx:end_idx], axis=1)
    return avg_fweight

def _variable_with_weight_decay( shape, stddev, wd, decoder=False):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _add_wd_and_summary( var, wd, collection_name=None):
    if collection_name is None:
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _bias_variable( shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='biases', shape=shape,
                          initializer=initializer)
    _variable_summaries(var)
    return var

def get_fc_weight_reshape( name, shape, num_classes=None):
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % shape)
    weights = data_dict[name][0]
    weights = weights.reshape(shape)
    if num_classes is not None:
        weights = _summary_reshape(weights, shape,
                                        num_new=num_classes)
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)
