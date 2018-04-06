import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as tic
import copy

activations = {
    1: 'relu',
    2: 'selu',
    3: 'tanh',
    4: 'sigmoid'
}


def show_examples(x, y):
    plt.figure(figsize=(5, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y[i]))


def norm_shift(count, high=3, stddev=1):
    '''
    Get a shift from a normal distribution. All shifts have an absolute value 1->high. Values are
    chosen from a normal distribution with mean 0 and the input stddev then rounded up to the nearest
    absolute value (ie: 0.5 -> 1 and -0.5 -> -1).
    count -- the number of shifts to return
    high -- clipped maximum, any value from the normal distribution with an absolute value > high
            is clipped to +/- high
    stddev -- the stddev of the normal distribution around 0, does not affect high
    Returns: 1d array of size count with possible values [1, high] both inclusive
    '''
    x = np.random.normal(loc=0, scale=1, size=count).astype(np.float32)
    x_new = (np.clip(np.ceil(np.absolute(x)), 1, high)
             * np.sign(x)).astype(np.int32)
    return x_new


class Gene(object):
    '''Genetic details for each element of the population
    '''

    def __init__(self, config={}):
        if 'num_layers' in config:
            self.num_layers = config["num_layers"]
        else:
            self.num_layers = 2
        if 'mutate_prob' in config:
            self.mutate_prob = config['mutate_prob']
        else:
            self.mutate_prob = 0.001

        self.fitness = -1
        # Size of hidden layers, can be 50->1000 by 50
        self.h = np.random.randint(low=1, high=21, size=(
            self.num_layers), dtype=np.int32) * 50
        # Enumerated activation functions, see get_func(a)
        indices = np.random.randint(low=1, high=5, size=(self.num_layers))
        self.a = [activations[i] for i in indices]
        # Dropout rate
        self.d = np.round(np.random.uniform(0.1, 1, size=self.num_layers), 2)
        # Batch size
        self.b = np.random.randint(low=1, high=7, dtype=np.int32)

    def __call__(self):
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout
        model = Sequential()
        model.add(Dense(self.h[0], activation=self.a[0],
                        input_shape=(784,), name="Dense1"))
        model.add(Dropout(self.d[0], name="Drop1"))
        for i in range(1, self.num_layers):
            model.add(Dense(self.h[i], activation=self.a[i], input_shape=(
                784,), name="Dense" + str(i + 1)))
            model.add(Dropout(self.d[i], name="Drop" + str(i + 1)))
        model.add(Dense(10, activation='softmax', name="output"))
        return model

    def __str__(self):
        output = []
        output.append(
            '_________________________________________________________________')
        output.append('input\t[size: {:{align}{width}}'.format(784, align='<', width='7') +
                      'batch size: {:{align}{width}}]'.format(2**self.b, align='<', width='25'))
        for i in range(self.num_layers):
            output.append('layer {}\t[size: {:{align}{width}}'.format(i + 1, self.h[i], align='<', width='7') +
                          'activation: {:{align}{width}}drop%: {:.2f}]'.format(self.a[i], self.d[i], align='<', width='14'))
        output.append('output\t[size: {:{align}{width}}'.format(10, align='<', width='7') +
                      'activation: {:{align}{width}}]'.format('softmax', align='<', width='25'))
        output.append(
            '_________________________________________________________________')
        return '\n'.join(output)

    def mutate(self):
        h_shift = norm_shift(self.num_layers) * np.random.binomial(1,
                                                                   self.mutate_prob, self.num_layers) * 50
        self.h = np.clip(self.h + h_shift, 50, 1000)
        a_shift = np.random.binomial(1, self.mutate_prob, self.num_layers)
        for i in range(self.num_layers):
            if a_shift[i]:
                self.a[i] = activations[np.random.randint(low=1, high=5)]
        d_shift = np.random.normal(loc=0, scale=0.1, size=self.num_layers) * \
            np.random.binomial(1, self.mutate_prob, self.num_layers)
        self.d = np.clip(self.d + d_shift, 0.1, 1)
        if np.random.binomial(1, self.mutate_prob):
            self.b = np.clip(self.b + norm_shift(self.num_layers), 1, 6)

    def cross(self, other):
        cross_mask = np.random.binomial(1, 0.5, (4, 3))
        new_gene = copy.copy(self)
        new_gene.fitness = -1
        for i in range(self.num_layers):
            if cross_mask[0, i]:
                new_gene.a[i] = other.a[i]
            if cross_mask[1, i]:
                new_gene.d[i] = other.d[i]
            if cross_mask[2, i]:
                new_gene.h[i] = other.h[i]
        if cross_mask[3, 0]:
            new_gene.b = other.b
        return new_gene

    def batch_size(self):
        return 2**self.b


class Population(object):

    def __init__(self, count, gene_config={}):
        from keras.datasets import mnist
        from keras.utils import np_utils
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print("X train/test shapes: {} / {}".format(x_train.shape,
                                                    x_test.shape))
        print("Y train/test shapes: {} / {}".format(y_train.shape,
                                                    y_test.shape))
        self.x_train = x_train.reshape(
            x_train.shape[0], 784).astype('float32') / 255
        self.x_test = x_test.reshape(
            x_test.shape[0], 784).astype('float32') / 255
        self.y_train = np_utils.to_categorical(y_train, 10)
        self.y_test = np_utils.to_categorical(y_test, 10)

        self.population = []
        self.pop_size = count
        for i in range(count):
            self.population.append(Gene(config=gene_config))

    def evolve(self, generations=1, model_epochs=10, elites=1, verbose=2):
        log = "Generation,Fitness,TrainingTime\n"
        for generation in range(generations):
            epoch_start = tic()
            for i in range(self.pop_size):
                if(self.population[i].fitness == -1):
                    if(verbose >= 1):
                        print("Training model {}...".format(i + 1))
                        print(self.population[i])
                    training_start = tic()
                    self._get_fitness(i, epochs=model_epochs, verbose=verbose)
                    log += '{},{:.4f},{:.4f}\n'.format(
                        generation, self.population[i].fitness,
                        tic() - training_start)
                else:
                    if(verbose >= 1):
                        print("Model {} already trained".format(i + 1))
                        log += '{},{:.4f},{:.4f}\n'.format(
                            generation, self.population[i].fitness,
                            tic() - training_start)
            self.population = sorted(
                self.population, key=lambda x: x.fitness, reverse=True)
            if(verbose >= 1):
                print("Best fitness for Generation {}: {:.4f}".format(
                    generation + 1, self.population[0].fitness))
            probs = np.array([gene.fitness for gene in self.population])
            total = probs.sum()
            probs = probs / total
            new_pop = self.population[:elites]  # Keep the most fit individuals
            for i in range(elites, self.pop_size):
                a, b = np.random.choice(
                    self.pop_size, size=2, replace=True, p=probs)
                child = self.population[a].cross(self.population[b])
                child.mutate()
                new_pop.append(child)
            self.population = new_pop
            if(verbose >= 1):
                print("Generation {} duration: {:.4f}".format(
                    i + 1, tic() - epoch_start))
        if(verbose >= 1):
            print("Training final generation...")
        for i in range(self.pop_size):
            if(self.population[i].fitness == -1):
                if(verbose >= 1):
                    print("Training model {}...".format(i + 1))
                    print(self.population[i])
                self._get_fitness(i, epochs=model_epochs, verbose=verbose)
                log += '{},{:.4f},{:.4f}\n'.format(
                    generations, self.population[i].fitness,
                    tic() - training_start)
            else:
                if(verbose >= 1):
                    print("Model {} already trained".format(i + 1))
                    log += '{},{:.4f},{:.4f}\n'.format(
                        generations, self.population[i].fitness,
                        tic() - training_start)

        self.population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True)
        output_log = open("output_log.csv", 'w')
        output_log.write(log)
        output_log.close()
        if(verbose >= 1):
            print("Final best fitness: {:.4f}".format(
                self.population[0].fitness))

    def _get_fitness(self, i, epochs=10, logging=False, verbose=1):
        from keras import metrics
        from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
        from keras import backend as K
        fit_verbose = max(verbose - 1, 0)
        model = self.population[i]()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=[metrics.categorical_accuracy])
        callback_list = [
            ModelCheckpoint("best_accuracy.hdf5",
                            monitor="val_categorical_accuracy",
                            save_best_only=True, mode='max'),
            CSVLogger("training.log"),
            TensorBoard()
        ] if logging else []
        if(verbose >= 1):
            print("Fitting Gene Model #{}".format(i + 1))
        history_callback = model.fit(self.x_train, self.y_train,
                                     batch_size=self.population[i].batch_size(),
                                     epochs=epochs,
                                     verbose=fit_verbose, validation_split=0.1,
                                     callbacks=callback_list)
        if logging:
            loss_history = history_callback.history["loss"]
            np.savetxt("loss_history.txt", np.array(
                loss_history), delimiter=",")
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        # TODO: Add punishment for time to train/model complexity
        self.population[i].fitness = score[1]

        # Must clear memory after each model to prevent slowdown
        K.clear_session()

    def best(self):
        if(self.population[0].fitness == 0):
            print("Warning: Population needs to be evolved.")
            return None
        return self.population[0]


def main():
    print("Main")


if __name__ == '__main__':
    np.random.seed(8675309)
    main()
