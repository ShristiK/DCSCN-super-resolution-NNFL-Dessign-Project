import logging

import tensorflow as tf

import DCSCN
import helper as util #import args, utilty as util

util.flags.DEFINE_boolean("save_results", False, "Save result, bicubic and loss images")

FLAGS = util.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    # modifying process/build options for faster processing
    if FLAGS.load_model_name == "":
        FLAGS.load_model_name = "default"
    FLAGS.save_loss = False
    FLAGS.save_weights = False
    FLAGS.save_images = False

    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_summary_saver()
    model.init_all_variables()

    logging.info("evaluate model performance")

    if FLAGS.test_dataset == "all":
        test_list = ['DIV2K']
    else:
        test_list = [FLAGS.test_dataset]

    for i in range(FLAGS.tests):
        model.load_model(FLAGS.load_model_name, i, True if FLAGS.tests > 1 else False)
        for test_data in test_list:
            test(model, test_data)


def test(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = 0

    for filename in test_filenames:
        psnr, ssim = model.do_for_evaluate(filename, output_directory=FLAGS.output_dir, output=FLAGS.save_results)
        total_psnr += psnr
        total_ssim += ssim

    logging.info("\n=== Average [%s] PSNR:%f, SSIM:%f ===" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
