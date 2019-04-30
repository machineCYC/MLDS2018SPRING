import os
import tensorflow as tf

from src.constants import *


class Seq2seq():
    def __init__(self, dim_video_frame, dim_video_feature, num_rnn_layers, 
        num_rnn_nuits, vocab_size, learning_rate, max_gradient_norm, word2index, max_decoder_step):

        self.dim_video_frame = dim_video_frame # 80
        self.dim_video_feature = dim_video_feature # 4096

        self.num_rnn_layers = num_rnn_layers # 2
        self.num_rnn_nuits = num_rnn_nuits # 每個 RNN 有幾個神經元 1024
        # self.encoder_embedding_size = encoder_embedding_size # TODO:
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.max_gradient_norm = max_gradient_norm
        self.word2index = word2index
        self.max_decoder_step = max_decoder_step

        # TODO:
        # self.image_weight = tf.get_variable("image_weight",
        #                           shape=(self.dim_video_feature, self.num_rnn_nuits),
        #                           initializer=tf.truncated_normal_initializer(stddev= 0.02))
        # self.image_bias = tf.get_variable("image_bias",
        #                           shape=(self.num_rnn_nuits),
        #                           initializer=tf.constant_initializer())
        




        self.word_embedded = tf.get_variable("word_emdeded",
                                  shape=(self.vocab_size, self.num_rnn_nuits),
                                  initializer=tf.truncated_normal_initializer(stddev= 0.02))
        self.word_weight = tf.get_variable("word_weight",
                                  shape=(self.num_rnn_nuits, self.vocab_size),
                                  initializer=tf.truncated_normal_initializer(stddev= 0.02))
        self.word_bias = tf.get_variable("word_bias",
                                  shape=(self.vocab_size),
                                  initializer=tf.constant_initializer())
        self.saver = None
        self.build_model()
    
    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.GRUCell(self.num_rnn_nuits)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.output_keep_prob_placeholder, seed=416) # dropout between rnn
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_rnn_layers)])
        return cell

    def build_model(self):
        
        # Input
        self.encoder_input = tf.placeholder(tf.float32, [None, self.dim_video_frame, self.dim_video_feature], name="encoder_input") # batch_size, dim_video_frame, dim_video_feature

        self.decoder_input = tf.placeholder(tf.int32, [None, None], name="decoder_input") # batch_size, ?
        self.decoder_target = tf.placeholder(tf.int32, [None, None], name="decoder_target") # batch_size, ?
        
        self.encoder_input_length = tf.placeholder(tf.int32, [None], name="encoder_input_length")
        self.decoder_target_length = tf.placeholder(tf.int32, [None], name="decoder_targets_length")
        self.max_target_sequence_length = tf.reduce_max(self.decoder_target_length, name="max_target_len")

        # Hyper parameter
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.output_keep_prob_placeholder = tf.placeholder(tf.float32, name="output_keep_prob_placeholder")

        # Encoder
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            encoder_input_flatten = tf.reshape(
                self.encoder_input, [-1, self.dim_video_feature]) # batch_size*80 ,4096

            # encoder_input_embedded = tf.matmul(encoder_input_flatten, self.image_weight) + self.image_bias # batch_size*80, num_rnn_nuits
            
            encoder_input_embedded = tf.layers.dense(encoder_input_flatten, self.num_rnn_nuits, use_bias=True) # batch_size*80, num_rnn_nuits
            encoder_input_embedded = tf.reshape(
                encoder_input_embedded, [self.batch_size, self.dim_video_frame, self.num_rnn_nuits]) # batch_size, 80, num_rnn_nuits

            encoder_cells = self._create_rnn_cell()
            encoder_output, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cells, inputs=encoder_input_embedded, sequence_length=self.encoder_input_length, 
                dtype=tf.float32) # encoder_output:batch_size, 80, num_rnn_nuits
        
        # Decoder
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_input_embedded = tf.nn.embedding_lookup(self.word_embedded, self.decoder_input) # self.word_embedded:vocab_size, num_rnn_nuits
            
            decoder_output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=416))

            # Training Helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_input_embedded, sequence_length=self.decoder_target_length, 
                time_major=False, name="training_helper")
            
            # Training Decoder
            decoder_cells = self._create_rnn_cell()
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cells, helper=training_helper, initial_state=encoder_state, output_layer=decoder_output_layer)

            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True, 
                maximum_iterations=self.max_target_sequence_length) # decoder_output:batch_size, ?, vocab_size
            
            self.decoder_logits = tf.identity(decoder_output.rnn_output) # batch_size, ?, vocab_size
            self.decoder_predict_train = tf.argmax(self.decoder_logits, axis=-1, name="decoder_pred_train") # batch_size, ?

        # Loss
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.one_hot(self.decoder_target, depth=self.vocab_size, dtype=tf.float32),
                    logits=self.decoder_logits) # batch_size, vocab_size

            self.total_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, axis=1))
        
            # Training summary for the current batch_loss
            tf.summary.scalar("Loss", self.total_loss)
            self.summary_op = tf.summary.merge_all()

        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.total_loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_step = optimizer.apply_gradients(zip(clip_gradients, trainable_params))


        # For Inference
        start_tokens = tf.fill([self.batch_size], self.word2index[BOS_WORD])
        end_token = self.word2index[EOS_WORD]

        # Inference Helper
        inference_decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.word_embedded, 
            start_tokens=start_tokens, 
            end_token=end_token)
        
        # Inference Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells, 
            helper=inference_decoding_helper, 
            initial_state=encoder_state, 
            output_layer=decoder_output_layer)
        
        inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, 
                maximum_iterations=self.max_decoder_step)
        
        self.decoder_predict_index = tf.expand_dims(inference_decoder_outputs.sample_id, -1) # batch_size, ?, 1
        self.decoder_predict_logits = inference_decoder_outputs.rnn_output # batch_size, ?, vocab_size


    def save_model(self, sess, model_file, step):
        print("----- Saving Model -----")
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=5)
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        self.saver.save(sess, model_file, global_step=step)

    def train(self, sess, encoder_input, encoder_input_length, 
        decoder_input, decoder_target, decoder_target_length, 
        output_keep_prob_placeholder):
        feed_dict = {self.encoder_input: encoder_input,
                     self.encoder_input_length: encoder_input_length,
                     self.decoder_input: decoder_input,
                     self.decoder_target: decoder_target,
                     self.decoder_target_length: decoder_target_length,
                     self.batch_size: len(encoder_input),
                     self.output_keep_prob_placeholder: output_keep_prob_placeholder}
        _, loss, summary = sess.run([self.train_step, self.total_loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary
    # FIXME:
    # def valid(self, sess, encoder_input, decoder_input, decoder_target):
    #     feed_dict = {self.encoder_input: encoder_input,
    #                   self.decoder_input: decoder_input,
    #                   self.decoder_target: decoder_target,
    #                   self.batch_size: len(encoder_input),
    #                   self.output_keep_prob_placeholder: 1.0}
    #     _, loss, summary = sess.run([self.total_loss, self.summary_op], feed_dict=feed_dict)
    #     return loss, summary

    def inference(self, sess, encoder_input, encoder_input_length):
        feed_dict = {self.encoder_input: encoder_input,
                     self.encoder_input_length: encoder_input_length,
                     self.batch_size: len(encoder_input),
                     self.output_keep_prob_placeholder: 1.0,}
        predict_index, logits = sess.run([self.decoder_predict_index, self.decoder_predict_logits], feed_dict=feed_dict)
        return predict_index, logits