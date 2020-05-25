import tensorflow as tf

import DCSCN
import helper as util

util.flags.DEFINE_string("file", "image.jpg", "Target filename")

FLAGS = util.get()


def main(_):
    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

    model.do_for_file(FLAGS.file, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
