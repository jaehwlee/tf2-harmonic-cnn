import time
import datetime

import os
from tqdm import tqdm
import tensorflow as tf
from data_loader.train_loader import TrainLoader
from data_loader.mtat_loader import DataLoader
from model import Model

# select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# fix random seed
SEED = 42
tf.random.set_seed(SEED)

# define epochs
STAGE1 = 60
STAGE2 = 200



# define loss and metrics
bce_loss = tf.keras.losses.BinaryCrossentropy()
train_auc = tf.keras.metrics.AUC()
train_loss = tf.keras.metrics.Mean()
valid_loss = tf.keras.metrics.Mean()
valid_auc = tf.keras.metrics.AUC()

test_loss = tf.keras.metrics.Mean()
test_auc = tf.keras.metrics.AUC()

# start time
start_time = time.time()

stage1_test_template = "AELoss : {:.5f}"
stage1_template = "Epoch: {}, TotalLoss : {:.5f},  AELoss : {:.5f}, KLLoss : {:.5f}"
test_template = "Test Loss : {}, Test AUC : {:.2f}%"
valid_template = "\nEpoch: {}, Valid Loss: {:.5f}, Valid AUC: {:.2f}%"
stage2_template = "Epoch : {}, Loss : {:.5f}, AUC : {:.2f}%"


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
test_log_dir = "logs/gradient_tape/" + current_time + "/test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
sgd2 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model=Model()

def load_data(root="../../semantic-tagging/dataset"):
    train_data = TrainLoader(root=root, split="train")
    valid_data = DataLoader(root=root, split="valid")
    test_data = DataLoader(root=root, split="test")
    return train_data, valid_data, test_data




#@tf.function
def adam_stage2_train_step(wave, labels):
    with tf.GradientTape() as tape:

        predictions = model(wave, training=True)
        loss = bce_loss(labels, predictions)

    train_variable = model.trainable_variables
    gradients = tape.gradient(loss, train_variable)

    adam.apply_gradients(zip(gradients, train_variable))

    train_loss(loss)
    train_auc(labels, predictions)


#@tf.function
def sgd_stage2_train_step(wave, labels):
    with tf.GradientTape() as tape:

        predictions= model(wave, training=True)
        loss = bce_loss(labels, predictions)

    train_variable = model.trainable_variables
    gradients = tape.gradient(loss, train_variable)

    sgd2.apply_gradients(zip(gradients, train_variable))

    train_loss(loss)
    train_auc(labels, predictions)



#@tf.function
def stage2_test_step(wave, labels):
    predictions = model(wave, training=False)

    loss = bce_loss(labels, predictions)
    valid_loss(loss)
    valid_auc(labels, predictions)


def stage2_train_adam(epochs):
    for epoch in range(epochs):
        for wave, labels in tqdm(train_ds):
            adam_stage2_train_step(wave, labels)

        stage2_log = stage2_template.format(
            epoch + 1, train_loss.result(), train_auc.result()*100
        )
        print(stage2_log)

    if (epoch % 19 == 0 and epoch!=0):
        for valid_wave, valid_labels in tqdm(valid_ds):
            stage2_test_step(valid_wave, valid_labels)
        valid_log = valid_template.format(epoch+1, valid_loss.result(), valid_auc.result()*100)
        print(valid_log)


def stage2_train_sgd(epochs):
    for epoch in range(epochs):
        for wave, labels in tqdm(train_ds):
            sgd_stage2_train_step(wave, labels)

        stage2_log = stage2_template.format(
            epoch + 1, train_loss.result(), train_auc.result()*100
        )
        print(stage2_log)
    if (epoch % 19 == 0 and epoch!=0):
        for valid_wave, valid_labels in tqdm(valid_ds):
            stage2_test_step(valid_wave, valid_labels)
        valid_log = valid_template.format(epoch+1, valid_loss.result(), valid_auc.result()*100)
        print(valid_log)





# load data
train_ds, valid_ds, test_ds = load_data()


print("\n\n@@@@@@@@@@@@@@@@@@@Start training Stage 2@@@@@@@@@@@@@@@@@@\n")
for i in range(4):
    if i == 0:
        epochs = 60
        stage2_train_adam(epochs)
    elif i == 1:
        epochs = 20
        stage2_train_sgd(epochs)
    elif i ==2:
        epochs = 20
        new_lr = 0.0001
        sgd2.lr.assign(new_lr)
        stage2_train_sgd(epochs)
    else:
        epochs= 100
        new_lr = 0.00001
        sgd2.lr.assign(new_lr)
        stage2_train_sgd(epochs)


"""
# save model
tf.keras.models.save_model(rese, "./tmp/gpu0_rese/")
tf.keras.models.save_model(classifier, "./tmp/gpu0_classifier/")
"""
# test
for wave, labels in tqdm(test_ds):
    stage2_test_step(wave, labels)

print("Time taken : ", time.time() - start_time)

test_result = test_template.format(valid_loss.result(), valid_auc.result() * 100)
print(test_result)
