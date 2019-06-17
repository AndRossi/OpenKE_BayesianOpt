import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
#Input training files from benchmarks/FB15K/ folder.


# batches_per_epoch: from 50 to 200 (int)
# learning rate: from 0.1 to 0.0001 (real)
# embedding_size: from 50 to 300 (int)
# ent_neg_rate: from 1 to 10 (int)
batches_per_epoch_range = []

def train_and_evaluate(batches_per_epoch=100, learning_rate=0.001, embedding_size = 100, ent_neg_rate = 1):
    con = config.Config()
    # True: Input test files from the same folder.
    con.set_in_path("./benchmarks/FB15K/")

    con.set_test_link_prediction(True)
    con.set_test_triple_classification(False)
    con.set_work_threads(8)
    con.set_train_times(10)
    con.set_nbatches(batches_per_epoch)
    con.set_alpha(learning_rate)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(embedding_size)
    con.set_ent_neg_rate(ent_neg_rate)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")
    con.set_log_on(True)

    #Models will be exported via tf.Saver() automatically.
    con.set_export_files("./res/model.vec.tf", 0)
    #Model parameters will be exported to json files automatically.
    con.set_out_files("./res/embedding.vec.json")
    #Initialize experimental settings.
    con.init()
    #Set the knowledge embedding model
    con.set_model(models.TransE)
    #Train the model.
    #con.run()
    #To test models after training needs "set_test_flag(True)".
    return con.test()

train_and_evaluate()

