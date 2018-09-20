
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.resnet import resnet_model
from official.resnet import resnet_run_loop

_HEIGHT = 64
_WIDTH = 64
_NUM_CHANNELS = 3
_NUM_CLASSES = 30


_NUM_IMAGES = {
    'train': 30494,
    'validation': 7727
}


DATASET_NAME = 'zj3'

def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


import pandas as pd
import skimage.io
import numpy as np
import os

root_path = u"../"
path_a = os.path.join(root_path, u"DatasetA_20180813")
path_b = os.path.join(root_path, u"DatasetB_20180919")


train_list    = pd.read_csv(os.path.join(root_path, "src/train.txt"))
validate_list = pd.read_csv(os.path.join(root_path, "src/train.txt"))
predict_list =  pd.read_csv(os.path.join(root_path, "src/train.txt"))
attributes    = pd.read_csv(os.path.join(root_path, "src/attributes.txt"))

#print(img_attr.shape)
attribute_matrix = np.array(attributes.iloc[:, 1:31]).astype(np.float32)

train_img_dict = {}
#_NUM_CLASSES=230
def generate_training():
  for i in range(train_list.shape[0]):
    if train_img_dict.get(i) is None:
      filename,tset,index = train_list.loc[i,["file","set","index"]]
      if tset=="A":
        filepath= path_a
      else:
        filepath= path_b
      img = skimage.io.imread(os.path.join(filepath, "train", filename))
      label = attribute_matrix[index,:]
      train_img_dict[i] = (img,label)
    else:
      img,label = train_img_dict[i]


    if img.shape == (64,64):
      #img = np.stack([img,img,img], axis=2)
      continue
    
    yield img, label


validate_img_dict = {}
def generate_validating():
  for i in range(validate_list.shape[0]):
    if validate_img_dict.get(i) is None:
      filename,tset,index = validate_list.loc[i,["file","set","index"]]
      if tset=="A":
        filepath= path_a
      else:
        filepath= path_b
      img = skimage.io.imread(os.path.join(filepath, "train", filename))
      label = attribute_matrix[index,:]
      validate_img_dict[i] = (img,label)
    else:
      img,label = validate_img_dict[i]


    if img.shape == (64,64):
      #img = np.stack([img,img,img], axis=2)
      continue
    
    yield img, label

def generate_predict():
  for i in range(predict_list.shape[0]):
      print(i,end=",")
      filename,tset = predict_list.loc[i,["file","set"]]
      if tset=="A":
        filepath= path_a
      else:
        filepath= path_b
      img = skimage.io.imread(os.path.join(filepath, "train", filename))
      
      if img.shape == (64,64):
        img = np.stack([img,img,img], axis=2)

      yield img,filename

    

def input_fn(mode, data_dir, batch_size, num_epochs=1, num_gpus=0, dtype=0):
  if mode=="train":
    dataset = tf.data.Dataset.from_generator(generate_training, 
                              (tf.uint8, tf.float32), 
                              (tf.TensorShape([64,64,3]), tf.TensorShape([30])))
    
    dataset = dataset.prefetch(buffer_size=batch_size * 8)
    dataset = dataset.shuffle(buffer_size=batch_size * 16)#_NUM_IMAGES['train'])
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda img,label: (preprocess_image(img, True),label),
          batch_size=batch_size,
          num_parallel_batches=1))

    
    dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    
    return dataset
  elif mode=="validate":
    dataset = tf.data.Dataset.from_generator(generate_validating, 
                              (tf.uint8, tf.float32), 
                              (tf.TensorShape([64,64,3]), tf.TensorShape([30])))
    
    dataset = dataset.prefetch(buffer_size=batch_size * 8)
    #dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda img,label: (preprocess_image(img, False),label),
          batch_size=batch_size,
          num_parallel_batches=1))
    dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

  elif mode=="predict":
    dataset = tf.data.Dataset.from_generator(generate_predict, 
                              (tf.uint8, tf.string), 
                              (tf.TensorShape([64,64,3]), tf.TensorShape([])))
    
    dataset = dataset.prefetch(buffer_size=batch_size * 8)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda img,tag: (preprocess_image(img, False),tag),
          batch_size=batch_size,
          num_parallel_batches=1))
    dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

  else:
    raise("parameter of input_fun unknown")

###############################################################################
# Running the model
###############################################################################
class ZJModel(resnet_model.Model):
  """Model class with appropriate defaults for ZJ-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for ZJ-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(ZJModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=32,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def zj_model_fn(features, labels, mode, params):
  """Model function for ZJ."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
      decay_rates=[1, 0.1, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the ZJ-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  ret = resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ZJModel,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype']
  )

  
  return ret


def define_zj_flags():
  resnet_run_loop.define_resnet_flags()
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(model_dir='zj_model_3',
                          resnet_size='32',
                          train_epochs=250,
                          epochs_between_evals=1,
                          batch_size=128)


def run_zj(flags_obj):
  """Run ResNet ZJ-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = input_fn

  resnet_run_loop.resnet_main(
      flags_obj, zj_model_fn, input_function, DATASET_NAME,
      shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])


def main(_):
  run_zj(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_zj_flags()
  absl_app.run(main)
