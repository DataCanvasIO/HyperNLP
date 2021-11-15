import pandas as pd
import tensorflow as tf
import torch

from utils.gpu_status import is_gpu_available
from hypernlp.config import Config


def train_tf(model,
          paths,
          search_space,
          epochs,
          batch_size,
          train_data,
          validate_data,
          param,
          optimizer,
          with_pretrained_model=None):

    if with_pretrained_model is not None:
        print('Initializing model...')
        model.load_weights(with_pretrained_model)
        print('Load weights from: ', with_pretrained_model)
        print('Initialization finished.')
    else:
        print("Without initialization model!")

    ## Training and validation
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # set tf logging detail display
    # save the model bert_pretrained
    # save weight but not model at the end of every 1 epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=paths.get("checkpoint_path"),
                                                             save_weights_only=True,
                                                             verbose=1, period=1)
    # stop train when there is no progress in 1 consecutive epochs
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=1)
    # lr decay
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=1, min_lr=1e-5)
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=paths.get("log_path"))
    print('Start training...')
    history = model.fit(x={'input_ids': train_data.get_full_data()[0], 'attention_mask': train_data.get_full_data()[1],
                           'token_type_ids': train_data.get_full_data()[2]},
                        y=train_data.get_full_data()[3],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_batch_size=batch_size,
                        validation_data=([validate_data.get_full_data()[0], validate_data.get_full_data()[1],
                                          validate_data.get_full_data()[2]], validate_data.get_full_data()[3]),
                        callbacks=[checkpoint_callback, reduce_lr_callback, tb_callback])
    print('Training finished!')

    print('Saving training history to csv...')
    hist_df = pd.DataFrame(history.history)
    with open(paths.get("history_path"), mode='w') as f:
        hist_df.to_csv(f, index=False)
    print('Saving finished.')


def train_pt(model,
          paths,
          search_space,
          epochs,
          batch_size,
          train_data,
          validate_data,
          param,
          optimizer,
          with_pretrained_model=None):

    if with_pretrained_model is not None:
        print('Initializing model...')
        if is_gpu_available() is True:
            model.load_state_dict(torch.load(with_pretrained_model))
        else:
            model.load_state_dict(torch.load(with_pretrained_model, map_location=lambda storage, loc: storage))
        print('Load weights from: ', with_pretrained_model)
        print('Initialization finished.')
    else:
        print("Without initialization model!")



