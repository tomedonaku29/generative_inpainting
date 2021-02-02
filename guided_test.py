import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--img_file', default='', type=str,
    help='The file of image to be processed')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    FLAGS = ng.Config('inpaint.yml')
    FLAGS.guided = True
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    t = time.time()
    # for i in range(100):
    image = 'input/' + args.img_file
    mask = image[:-4] + '_mask.png'
    out = 'output/' + args.img_file
    base = os.path.basename(mask)

    guidance = cv2.imread(image[:-4] + '_edge.jpg')
    image = cv2.imread(image)
    mask = cv2.imread(mask)

    assert image.shape == mask.shape
    assert guidance.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    guidance = guidance[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, image.shape[0], image.shape[1]*3, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    image = np.expand_dims(image, 0)
    guidance = np.expand_dims(guidance, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, guidance, mask], axis=2)

    # load pretrained model
    result = sess.run(output, feed_dict={input_image_ph: input_image})
    print('Processed: {}'.format(out))
    cv2.imwrite(out, result[0][:, :, ::-1])

    print('Time total: {}'.format(time.time() - t))
