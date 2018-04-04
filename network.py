# %% Imports
import numpy as np
from timeit import default_timer as timer
from generators import SampledGenerator, deltaIT_unnorm
# from keras.layers.normalization import BatchNormalization
# from keras.utils import to_categorical, plot_model


def log_gpu():
    import subprocess
    # This line outputs the first 2 minutes of GPU performance to GPU-stats.log
    subprocess.Popen(
        "timeout 120 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > ./GPU-stats.log", shell=True)
    # These lines show the table of the output. Only run after
    # 120 seconds have passed since the previous line.
    # gpu = pd.read_csv("./GPU-stats.log")
    # gpu.plot()
    # plt.show()


def lamb(x):
    import tensorflow as tf
    return tf.concat([x, x[:, :, :20]], axis=2)


# %% Model
def build_model():
    from keras.models import Model
    from keras.layers import Dense, Dropout, Input, concatenate
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Lambda
    image_in = Input(shape=(150, 1440, 1), name="image_input")
    polar1 = Lambda(lambda x: lamb(x), name="polar_wrapper_1")(image_in)
    conv1 = Conv2D(64, (3, 3), activation='relu', name="image_conv1")(polar1)
    drop1 = Dropout(0.2)(conv1)
    max1 = MaxPooling2D(2, 2)(drop1)
    polar2 = Lambda(lambda x: lamb(x), name="polar_wrapper_2")(max1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name="image_conv2")(polar2)
    drop2 = Dropout(0.2)(conv2)
    max2 = MaxPooling2D(2, 2)(drop2)
    polar3 = Lambda(lambda x: lamb(x), name="polar_wrapper_3")(max2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name="image_conv3")(polar3)
    drop3 = Dropout(0.2)(conv3)
    max3 = MaxPooling2D(2, 2)(drop3)
    polar4 = Lambda(lambda x: lamb(x), name="polar_wrapper_4")(max3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name="image_conv4")(polar4)
    drop4 = Dropout(0.2)(conv4)
    max4 = MaxPooling2D(2, 2)(drop4)
    polar5 = Lambda(lambda x: lamb(x), name="polar_wrapper_5")(max4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name="image_conv5")(polar5)
    drop5 = Dropout(0.2)(conv5)
    max6 = MaxPooling2D(2, 2)(drop5)
    flat = Flatten()(max6)
    dense = Dense(50, activation='relu', name="image_dense")(flat)
    drop6 = Dropout(0.2)(dense)

    seg_in = Input(shape=(360,), name="segmentation_input")
    dense1 = Dense(180, activation='relu', name="seg_dense1")(seg_in)
    seg_drop1 = Dropout(0.2)(dense1)
    dense2 = Dense(180, activation='relu', name="seg_dense2")(seg_drop1)
    seg_drop2 = Dropout(0.2)(dense2)
    dense3 = Dense(50, activation='relu', name="seg_dense3")(seg_drop2)
    seg_drop3 = Dropout(0.2)(dense3)

    merged = concatenate([drop6, seg_drop3])

    merged_dense = Dense(50, activation='relu', name="merged_dense")(merged)
    merge_drop = Dropout(0.2)(merged_dense)
    output = Dense(1, name="Output")(merge_drop)

    model = Model(inputs=[image_in, seg_in], outputs=output)
    return model


def train_split(X, Y, split=0.2):
    split_index = int(X.shape[0] * split)
    x_test, x_train = np.split(X, [split_index])
    y_test, y_train = np.split(Y, [split_index])
    return x_train, x_test, y_train, y_test


def split_data(X, X_seg, Y, split=0.2):
    idx = np.random.permutation(X.shape[0])
    split_index = int(idx.shape[0] * split)
    test_idx = idx[:split_index]
    train_idx = idx[split_index:]
    X_test = X[test_idx]
    X_train = X[train_idx]
    X_seg_test = X_seg[test_idx]
    X_seg_train = X_seg[train_idx]
    Y_test = Y[test_idx]
    Y_train = Y[train_idx]
    return X_train, X_test, X_seg_train, X_seg_test, Y_train, Y_test


def percent_acc(y_true, y_pred):
    import tensorflow as tf
    return tf.reduce_mean(np.absolute((y_pred / y_true) - 1))


# %% Main
def main():
    from keras.callbacks import ModelCheckpoint, CSVLogger
    from keras.utils.vis_utils import plot_model
    np.random.seed(SampledGenerator.SEED)

    model_name = "resampled_model_1"
    # Data when used with the generator
    data = SampledGenerator(batch_size=16, useSegmentation=True, rotations=4)
    data.split_data(training=0.7, validation=0.1, test=0.2)
    normalizer = data.normalizerY
    model = build_model()
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    model.summary()
    plot_model(model, to_file=model_name + "/model_plot.png", show_shapes=True)
    start = timer()
    epochs = 10
    callbacks_list = [
        ModelCheckpoint(model_name + "/best_error.hdf5",
                        monitor="val_mean_absolute_error",
                        save_best_only=True, mode='min'),
        CSVLogger("training.log"),
        # TensorBoard(log_dir='./resampled_model_1/logs', histogram_freq=1, )
    ]
    history_callback = model.fit_generator(
        data.generate(training=True),
        data.batch_count(training=True),
        epochs=epochs,
        validation_data=data.generate(validation=True),
        validation_steps=data.batch_count(validation=True),
        shuffle=False,
        callbacks=callbacks_list
    )
    # history_callback = model.fit([X_train, X_seg_train], Y_train,
    #                              callbacks=callbacks_list, batch_size=16,
    #                              epochs=10, validation_split=0.2)
    duration = timer() - start
    loss_history = history_callback.history["loss"]
    np.savetxt("loss_history.txt", np.array(loss_history), delimiter=",")
    print("Time to train {:.2f}".format(duration))
    model.save(model_name+'/model.hdf5')

    # Testing when used with generator
    over = [0, 0.0]
    under = [0, 0.0]
    exact = 0
    all_diff = []
    for X_test, y_test in data.generate(num_batches=1):
        predictions = model.predict_on_batch(X_test)
        diff = normalizer.denormalize(
            y_test) - normalizer.denormalize(predictions)
        for i in range(diff.shape[0]):
            if(diff[i] > 0):
                under = [under[0] + 1, under[1] + diff[i]]
            elif(diff < 0):
                over = [over[0] + 1, over[1] + diff[i]]
            else:
                exact += 1
            all_diff.append(diff)
    f = open(model_name+"/results.npy", "wb")
    np.save(file=f, arr=np.concatenate(all_diff))
    f.close()
    over_avg = deltaIT_unnorm(over[1] / over[0])
    under_avg = deltaIT_unnorm(under[1] / over[0])
    print("Over: {}, Avg over: {:.6f}\n" +
          "Under: {}, Avg under: {:.6f}\n" +
          "Exact: {}".format(over[0], over_avg, under[0], under_avg, exact))

    # predictions = model.predict([X_test, X_seg_test])
    # diff = Y_test - predictions
    # over = diff[diff > 0]
    # under = diff[diff < 0]
    # print("Over: {}, mean {:.4f}\n".format(over.shape[0], over.mean()) +
    #       "Under: {}, mean {:.4f}\n".format(under.shape[0], under.mean()) +
    #       "Total: {}".format(diff.shape[0]))
    # np.save("diff_sampled.npy", diff)


if __name__ == '__main__':
    main()
