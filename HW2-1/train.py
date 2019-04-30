import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime

from src.constants import *
from src.read import DataSets, read_word2index, read_index2word, read_caption, parse_index_sentence
from src.util import write_arguments_to_file
from src.bleu_eval import BLEU
from src.model import Seq2seq


def main(args):
    subdir = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
    model_dir = "./save_models/{}/model.ckpt".format(subdir)
    logs_dir = "./logs/{}".format(subdir)
    log_dir = os.path.expanduser(logs_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, "arguments.txt"))

    word2index = read_word2index(dic_full_path=os.path.join(args.DATA_DIR_PATH, "dictionary.txt"))
    index2word = read_index2word(dic_full_path=os.path.join(args.DATA_DIR_PATH, "dictionary.txt"))

    train_captions = read_caption(caption_full_path=os.path.join(args.DATA_DIR_PATH, "padding_training_label.json"))
    test_captions = read_caption(caption_full_path=os.path.join(args.DATA_DIR_PATH, "padding_testing_label.json"))

    train_data = DataSets(
        image_feature_dir_path=os.path.join(args.DATA_DIR_PATH, "training_data/feat"), 
        captions=train_captions)
    valid_data = DataSets(
        image_feature_dir_path=os.path.join(args.DATA_DIR_PATH, "testing_data/feat"), 
        captions=test_captions)

    graph = tf.Graph()
    with graph.as_default():
        train_model = Seq2seq(
            dim_video_frame=train_data._dim_video_frame, 
            dim_video_feature=train_data._dim_video_feature,
            num_rnn_layers=args.num_rnn_layers,
            num_rnn_nuits=args.num_rnn_nuits, 
            vocab_size=len(word2index), 
            learning_rate=args.learning_rate,
            max_gradient_norm=5,
            word2index=word2index,
            max_decoder_step=50
        )

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(graph=graph, 
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())

        t_summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        nbr_of_steps = train_data._nbr_of_caption // args.batch_size
        total_steps = 0
        for epoch in range(args.epochs):
            
            start_epoch_time = datetime.now()
            for steps in range(nbr_of_steps):

                total_steps += 1
                image_feature, image_caption, image_caption_len = train_data.next_batch(args.batch_size) # batch_size, 80, 4096


                total_loss, t_summary = train_model.train(sess=sess, encoder_input=image_feature, 
                    encoder_input_length=[train_data._dim_video_frame] * args.batch_size, 
                    decoder_input=image_caption[:, :-1], 
                    decoder_target=image_caption[:, 1:], 
                    decoder_target_length=image_caption_len,
                    output_keep_prob_placeholder=args.output_keep_prob)

                t_summary_writer.add_summary(t_summary, total_steps)

                if (total_steps % 100) == 0:
                    print("Epochs:{}, Steps:{}, Loss:{}".format(epoch+1, total_steps, total_loss))

                if (total_steps % args.save_steps) == 0:
                    train_model.save_model(sess, model_dir, total_steps)
            
            end_epoch_time = datetime.now()
            print("Epochs:{}, Steps:{}, Loss:{}, Times:{}sec/epoch".format(epoch+1, total_steps, 
                        total_loss, (end_epoch_time-start_epoch_time).seconds))

            # valid
            v_video_feature, v_video_id = valid_data.next_inference_batch(args.batch_size)
            inderence_caption_index, logits = train_model.inference(sess=sess, encoder_input=v_video_feature,
                encoder_input_length=[train_data._dim_video_frame] * args.batch_size)

            inference_captions = []
            for caption_index in inderence_caption_index:
                word_sentence = parse_index_sentence(caption_index, index2word)
                word_sentence = " ".join(word_sentence)
                print(word_sentence)
                inference_captions.append(word_sentence)
            
            # print(inderence_caption_index.shape, logits.shape)

            inference_result = {}
            for v_id, v_caption in zip(v_video_id, inference_captions):
                inference_result[v_id] = v_caption

            test_video_caption = read_caption(caption_full_path=os.path.join(args.DATA_DIR_PATH, "testing_label.json"))
            bleu=[]
            for item in test_video_caption:
                if item["id"] in inference_result:
                    score_per_video = []
                    captions = [x for x in item["caption"]]
                    score_per_video.append(BLEU(inference_result[item["id"]], captions, True))
                    bleu.append(score_per_video[0])
            average = sum(bleu) / len(bleu)

            v_summary = tf.Summary()
            v_summary.value.add(tag="validate/batch_avg_bleu", simple_value=average)
            t_summary_writer.add_summary(v_summary, epoch)
            print("Batch testing data average bleu score:{}".format(average))
        
        train_model.save_model(sess, model_dir, total_steps)

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "data/MLDS_hw2_1_data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DATA_DIR_PATH",
        type=str,
        default=DATA_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help=""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=""
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help=""
    )
    parser.add_argument(
        "--num_rnn_layers",
        type=int,
        default=2,
        help=""
    )
    parser.add_argument(
        "--output_keep_prob",
        type=float,
        default=0.6,
        help=""
    )
    parser.add_argument(
        "--num_rnn_nuits",
        type=int,
        default=1024,
        help=""
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help=""
    )
    main(parser.parse_args())