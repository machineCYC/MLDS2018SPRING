import os
import argparse
import numpy as np
import tensorflow as tf


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    save_model_path = os.path.join(args.SAVE_MODLE_DIR_PATH, args.MODEL_TYPES, args.MODEL_TYPES)
    if tf.gfile.Exists(save_model_path):
        tf.logging.error("path is not exists :{}".format(save_model_path))

    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_model_path + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(save_model_path)))

        # tensor_name = [n.name for n in tf.get_default_graph().as_graph_def().node]

        x_placeholder = tf.get_default_graph().get_tensor_by_name(args.INPUT_PLACEHOLDER_TENSOR_NAME)
        predict = tf.get_default_graph().get_tensor_by_name(args.OUTPUT_TENSOR_NAME)

        prediction = sess.run(predict, feed_dict={x_placeholder:np.linspace(0.0001, 1.0, num=1000)[:, np.newaxis]})
        
        predict_save_path = os.path.join(args.LOG_DIR_PATH, args.MODEL_TYPES, "predict_" + args.MODEL_TYPES + ".npy")
        np.save(predict_save_path, prediction)


if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")
    MODEL_TYPES = "shallow"
    INPUT_PLACEHOLDER_TENSOR_NAME = "Placeholder:0"
    OUTPUT_TENSOR_NAME = "dense_1/BiasAdd:0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--LOG_DIR_PATH",
        type=str,
        default=LOG_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--SAVE_MODLE_DIR_PATH",
        type=str,
        default=SAVE_MODLE_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--MODEL_TYPES",
        type=str,
        default=MODEL_TYPES,
        help=""
    )
    parser.add_argument(
        "--INPUT_PLACEHOLDER_TENSOR_NAME",
        type=str,
        default=INPUT_PLACEHOLDER_TENSOR_NAME,
        help=""
    )
    parser.add_argument(
        "--OUTPUT_TENSOR_NAME",
        type=str,
        default=OUTPUT_TENSOR_NAME,
        help=""
    )
    main(parser.parse_args())