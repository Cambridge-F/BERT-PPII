
import numpy as np

from keras.layers import Lambda, Dense
from keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
import csv
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.83)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

set_gelu('tanh')  # 切换gelu版本

num_classes = 2
maxlen = 21
batch_size = 128
config_path ='bert/cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert/cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert/cased_L-12_H-768_A-12/vocab.txt'


def load_data(filename,name):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    #
    with open(filename,encoding='utf-8') as f:
        i = 0
        for l in f:
            text, label = l.strip().split(',')
            D.append((text, int(label)))
            i = i+1
    print(i)
    print(name)
    return D

def write_data(test_data):
    L = test_data
    with open("datasets/WS13/WS13_test_data_3.csv",'w',encoding='utf-8') as f:
        j = 0
        writer = csv.writer(f)
        for row in L:
            writer.writerow(row)
            j = j + 1
    print(j)

# 加载数据集
train_data_1 = load_data('datasets/nonstrict/WS13/trainset/WS13_balance.csv','train')
np.random.seed(200)
np.random.shuffle(train_data_1)
train_data_2 = shuffle(train_data_1,random_state=1337)
train_data, valid_data = train_test_split(train_data_2, test_size=0.2, random_state=42)
#test_data,valid_data = train_test_split(test_valid_data,test_size=0.5,random_state=42)

#write_data(test_data)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    print("class_start")
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    print("class_end")




def textcnn(inputs,kernel_initializer):
    cnn1= keras.layers.Conv1D(
        256,
        3,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)
    cnn2 = keras.layers.Conv1D(
        256,
        4,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)
    cnn3 = keras.layers.Conv1D(
        256,
        5,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    output = keras.layers.concatenate(
        [cnn1,cnn2,cnn3],
        axis=-1
    )
    output = BatchNormalization()(output)
    #output = keras.layers.Dropout(0.5)(output)
    return output

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
print(type(output))         #shape = (batch_size,768)
print(output)


all_token_embedding = keras.layers.Lambda(lambda x: x[:,1:-1], name='all-token')(bert.model.output)
print(all_token_embedding)  #shape = (batch_size,sentence_leagth,768)

print("output.....")
cnn_features = textcnn(all_token_embedding,bert.initializer)  #shape=[batch_size,cnn_output_dim]
concat_features = keras.layers.concatenate([output,cnn_features],axis=-1)


dense = keras.layers.Dense(
    units=512,
    activation='relu',
    kernel_initializer=bert.initializer
    )(concat_features)
output = keras.layers.Dense(
    units=num_classes,
    activation='sigmoid',
    kernel_initializer=bert.initializer
    )(concat_features)



'''
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)
'''
print(type(bert.model.input))
#temp = BatchNormalization()(bert.model.input) ################################################################################################
#print(type(temp))
model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    #loss='categorical_crossentropy',
    #optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1,
    }),
    metrics=['accuracy'],
)

# 转换数据集
print("转换数据集")
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
#test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('nonstrict_model_demo13_20.weights')
        '''
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )
        '''

def show_acc_loss_plot(history):
    print(history.history)
    print(history.epoch)
    ##########################################
    print("Plot training & validation accuracy values")
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model acc')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test', 'Train_Loss', 'Test_Loss'], loc='upper left')
    plt.savefig("bert_demo13")
    plt.show()
    ##########################################
    '''保存训练日志'''
    pd.DataFrame(history.history).to_csv('bert_demo13_3_training_log.csv', index=False)


def show_roc_auc_plot(data):
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
    # ns_auc = roc_auc_score(y_true, ns_probs)
    lr_auc = roc_auc_score(y_true, y_pred)
    # summarize scores
    # print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    # ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)
    # plot the roc curve for the model
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='bert_demo4')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

if __name__ == '__main__':

    evaluator = Evaluator()

    myearlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                                  verbose=0, mode='min', baseline=None,restore_best_weights=False)

   


    #训练
    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=150,
        verbose=1,
        #class_weight= class_wight,
        callbacks=[evaluator],
        validation_steps=len(valid_generator),
        validation_data=(valid_generator.forfit())
    )
    #model.save('model13.h5')
    ##########################################
    '''保存训练日志'''
    #pd.DataFrame(history.history).to_csv('training_bert_demo5_log.csv', index=False)
    #show_acc_loss_plot(history)
    #model.save_weights('best_model.weights')

    model.load_weights('nonstrict_model_demo13_20.weights')
    #show_roc_auc_plot(test_generator)
    #print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:
    model.load_weights('nonstrict_model_demo13_20.weights')
