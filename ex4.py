import sys
from gcommand_loader import GCommandLoader
import torch.nn as nn
import torch

num_workers_train = 20
num_workers_validation = 20
num_workers_test = 20
batch_size_train = 100
batch_size_validation = 100
batch_size_test = 100
epochs_num = 8
learning_rate = 0.001
# convolution params
conv_kernel_size = 3
conv_stride = 1
conv_padding = 2
pool_kernel_size = 2
pool_stride = 2


class CNN(nn.Module):
    def __init__(self, train_loader, valid_loader, test_loader, test_set):
        """
        Data Members:
        ----------
        train_loader :  train data set loader.
        valid_loader :  validation data set loader.
        test_loader :   test data set loader.
        test_set :      test set.
        inputs_num :    number of inputs in train set.
        activation :    activation function: ReLU.
        pool :          max pool.
        dropout :       drop out to avoid over fitting.
        loss_func :     loss function: cross entropy.
        optimizer :     optimization algorithm: Adam.
        layer1 :        convolution layer no 1.
        layer2 :        convolution layer no 2.
        layer3 :        convolution layer no 3.
        fc1 :           fully connected layer no 1.
        fc2 :           fully connected layer no 2.
        fc3 :           fully connected layer no 3.
        """
        super(CNN, self).__init__()

        # all the necessary loaded data.
        self.train_loader = train_loader
        self.inputs_num = len(self.train_loader)
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.test_set = test_set

        # use reLu as an activation function.
        self.activation = nn.ReLU()
        # use max pool to detect structures.
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        # avoid over fitting by using dropout.
        self.dropout = nn.Dropout()

        # create convolution layers.
        conv1 = nn.Conv2d(1, 20, conv_kernel_size, conv_stride, conv_padding)
        self.layer1 = nn.Sequential(conv1, self.activation,  self.pool)
        conv2 = nn.Conv2d(20, 30, conv_kernel_size, conv_stride, conv_padding)
        self.layer2 = nn.Sequential(conv2, self.activation,  self.pool)
        conv3 = nn.Conv2d(30, 30, conv_kernel_size, conv_stride, conv_padding)
        self.layer3 = nn.Sequential(conv3, self.activation,  self.pool)

        # set fully connected layers.
        self.fc1 = nn.Linear(8820, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 30)

        # use cross entropy loss function.
        self.loss_func = nn.CrossEntropyLoss()
        # use Adam as optimization algorithm.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, data):
        """
        Operation of forward prop step.
        """
        outputs = self.layer3(self.layer2(self.layer1(data)))
        outputs = self.dropout(outputs.reshape(outputs.size(0), -1))
        return self.fc3(self.fc2(self.fc1(outputs)))

    def forward_prop(self, data, labels):
        """
        Forward propagation step.

        Parameters
        ----------
        data : current data.
        labels : correct labels.

        Returns
        -------
        Prediction and loss.
        """
        # call the "forward" method of module.
        cnn = self
        outputs = cnn(data)
        self = cnn
        # calculate loss.
        loss = self.loss_func(outputs, labels)
        return outputs, loss

    def backward_prop(self, loss):
        """
        Backward propagation step.

        Parameters
        ----------
        loss : current loss.

        Returns
        -------
        Loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_model(self):
        """
        Train the model using the train data set.
        """
        print("Model training has started.")
        for epoch in range(epochs_num):
            for (x, y) in self.train_loader:
                outputs, loss = self.forward_prop(x, y)
                self.backward_prop(loss)
            print("Finished training epoch number: {}.".format(epoch))

    def test(self):
        """
        Get predictions for validation dataset and print the accuracy.
        """
        # do not train with this data.
        self.train(False)
        # get predictions for validation set
        with torch.no_grad():
            # init counters to calculate accuracy.
            correct_predictions = 0
            total = 0
            for x, y in self.valid_loader:
                total += y.size(0)
                outputs = self(x)
                blank, y_hat = torch.max(outputs.data, 1)
                term = (y_hat == y)
                correct_predictions += term.sum().item()
            acc = (correct_predictions / total) * 100
            print('Model Accuracy: {} %'.format(acc))

    def get_final_predictions(self):
        """
        Get the final test predictions.

        Returns
        -------
        list of input names and list of predictions.
        """
        # extract all file names to list.
        inputs = []
        for x in self.test_set:
            inputs.append(extract_file_name(x[0]))
        i = 0
        # predict.
        predictions = []
        for (x, blank) in self.test_loader:
            outputs = self(x)
            blank, prediction = torch.max(outputs.data, 1)
            prediction = prediction.tolist()
            predictions.extend(prediction)
            i += 1
        return zip(inputs, predictions)

    def predict(self):
        """
        Use the trained model to give predictions to test set.
        Write predictions to text file named "test_y".
        """
        prediction_list = self.get_final_predictions()
        file = open("test_y", "w")
        for (x, y_hat) in prediction_list:
            file.write('{}, {}\n'.format(x, str(y_hat)))
        file.close()


def extract_file_name(file_path):
    """
    File path in format "name.wav".

    Parameters
    ----------
    file_path : train set path.

    Returns
    -------
    "name" without ".wav".
    """
    file_name = ""
    for i in range(0, len(file_path)):
        if file_path[len(file_path) - i - 1] == "\\" or file_path[len(file_path) - i - 1] == "/":
            break
        else:
            file_name += file_path[len(file_path) - i - 1]
    return file_name[::-1]


def load_data(train_path, validation_path, test_path):
    """
    Load the data sets using the data loader.

    Parameters
    ----------
    train_path : train set path.
    validation_path : validation set path.
    test_path : test set path.

    Returns
    -------
    Data arranged in lists.
    """
    # load the train set.
    train_set = GCommandLoader(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True,
        num_workers=num_workers_train, pin_memory=True, sampler=None)
    # load the validation set.
    validation_set = GCommandLoader(validation_path)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size_validation, shuffle=True,
        num_workers=num_workers_validation, pin_memory=True, sampler=None)
    # load the test set.
    test_set = GCommandLoader(test_path)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=None,
        num_workers=num_workers_test, pin_memory=True, sampler=None)
    return train_loader, validation_loader, test_loader, test_set


if __name__ == '__main__':
    train_path = sys.argv[1]
    validation_path = sys.argv[2]
    test_path = sys.argv[3]

    # load the data.
    train_loader, validation_loader, test_loader, test_set = load_data(train_path, validation_path, test_path)
    print("Finished loading data.")

    # create a CNN model, train it and then predict.
    model = CNN(train_loader, validation_loader, test_loader, test_set.spects)
    model.train_model()
    model.test()
    model.predict()
