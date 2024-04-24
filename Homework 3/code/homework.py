# Import libraries
import numpy as np
import pandas as pd
import copy
import random

"""
Definitions for utility functions
"""


def train_test_split(X, y=None, test_size=0.25, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    """

    if isinstance(X, pd.DataFrame):
        if y is not None and X.shape[0] != len(y):
            raise ValueError("y and X must have the same number of samples")
        X_index = X.index.tolist()  # Handle DataFrames with custom indices
    else:
        X_index = range(X.shape[0])  # Handle NumPy arrays

    random.seed(random_state)
    X_index = list(X_index)
    random.shuffle(X_index)

    if isinstance(test_size, int):
        test_size = float(test_size) / len(X_index)

    if test_size < 0 or test_size > 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")

    test_index = int(len(X_index) * test_size)
    X_train, X_test = X[X_index[:-test_index]], X[X_index[-test_index:]]

    if y is not None:
        y_train, y_test = y[X_index[:-test_index]], y[X_index[-test_index:]]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


class OneHotEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse
        self.encoders = {}
        self.feature_names_out = None

    def fit(self, X):
        if not isinstance(X, pd.Series):
            raise ValueError("Input must be a pandas Series")
        self.encoders = {
            col: pd.api.types.CategoricalDtype(categories=X.unique())
            for col in [X.name]
        }
        self.feature_names_out = [
            f"{X.name}_{cat}" for cat in self.encoders[X.name].categories
        ]
        return self

    def transform(self, X):
        if not isinstance(X, pd.Series):
            raise ValueError("Input must be a pandas Series")
        encoded_df = X.to_frame(name=X.name).astype(self.encoders)
        encoded_df = pd.get_dummies(encoded_df, sparse=self.sparse)
        return encoded_df.astype(int)

    def get_feature_names_out(self):
        if self.feature_names_out is None:
            raise ValueError("Encoder not fitted yet.")
        return self.feature_names_out


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for scaling.
        """
        self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            # Prevent division by zero
            self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        X_centered = X - self.mean_
        if self.with_std:
            X_scaled = X_centered / self.scale_
        else:
            X_scaled = X_centered
        return X_scaled


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Computes the accuracy score.
    """

    # Check if shapes match
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same size.")

    # Cast y_true and y_pred to float32
    y_true = np.array(y_true).astype(np.float32)
    y_pred = np.array(y_pred).astype(np.float32)

    # Count the number of correct predictions
    correct = np.sum(y_true == y_pred)

    # Return accuracy based on normalization parameter
    if normalize:
        return correct / len(y_true)
    else:
        return correct


def shuffle(X, y=None, random_state=None):
    """
    Shuffle the data in X and optionally y in unison.
    """
    if random_state is None:
        random_state = np.random.seed()

    permutation = np.random.permutation(len(X))

    shuffled_X = X[permutation]
    if y is not None:
        shuffled_y = y[permutation]
        return shuffled_X, shuffled_y
    else:
        return shuffled_X


# Define the fully-connected layer
class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(
            loc=0, scale=0.2, size=(input_size, output_size)
        )
        self.bias = np.random.normal(loc=0, scale=0.2, size=(1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# Define the activation function layer
class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)


# Define the relu activation function
def relu(x):
    return np.maximum(x, 0)


# Define the leaky relu activation function
def leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha * x)


# Define derivate of relu activation function
def relu_prime(x):
    return np.array(x >= 0).astype("int")


# Define the derivative of the leaky relu activation function
def leaky_relu_prime(x, alpha=0.01):
    return np.array(x > 0).astype("float") + alpha * np.array(x <= 0).astype("float")

# Define error function (mape)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Define derivative of error function (mape)
def mape_prime(y_true, y_pred):
    return -np.mean(np.sign(y_true - y_pred) / y_true) * 100


# Define base neural network class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Functions to add layers to the model
    def add(self, layer):
        self.layers.append(layer)

    # Initialize loss
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Function to make the prediction
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, initial_learning_rate, decay_rate):
        """
        Trains the model for a given number of epochs with learning rate decay.

        Args:
        - X: The training data.
        - y: The training labels.
        - epochs: The number of epochs to train for.
        - initial_learning_rate: The initial learning rate.
        - decay_rate: The learning rate decay rate.
        """

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        samples = len(x_train)
        err = None
        best_acc = -999
        best_model = None

        for epoch in range(epochs):
            # Shuffling within each epoch
            X_train_shuffled, y_train_shuffled = shuffle(
                x_train, y_train, random_state=epoch
            )

            learning_rate = initial_learning_rate * (decay_rate ** (epoch // 18))
            err = 0

            for j in range(samples):
                output = X_train_shuffled[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute the loss
                err += self.loss(y_train_shuffled[j], output)

                # Backward propagation
                error = self.loss_prime(y_train_shuffled[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # Compute average error on all sample
            err /= samples

            # Compute validation error
            pred = self.predict(x_val)
            pred = get_pred(pred)
            label = [ele[0][0] for ele in y_val]
            acc = accuracy_score(label, pred)

            # Checkpoint model
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self)

        return best_model, best_acc


# Function to load the data
def load_train_data():
    # Load the data
    train_data_path = "train_data.csv"
    train_label_path = "train_label.csv"
    test_data_path = "test_data.csv"
    df = pd.read_csv(train_data_path)
    labels = pd.read_csv(train_label_path)
    enc_df = pd.read_csv(test_data_path)

    # Create the Encoding DataFrame
    enc_df = pd.concat([df, enc_df])

    # Combine both DataFrames
    df["BEDS"] = labels["BEDS"]

    # Drop some columns
    cols_to_drop = [
        "BROKERTITLE",
        "ADDRESS",
        "STATE",
        "MAIN_ADDRESS",
        "ADMINISTRATIVE_AREA_LEVEL_2",
        "LOCALITY",
        "SUBLOCALITY",
        "STREET_NAME",
        "LONG_NAME",
        "FORMATTED_ADDRESS",
    ]
    df = df.drop(cols_to_drop, axis=1)
    enc_df = enc_df.drop(cols_to_drop, axis=1)

    # Encode the TYPE column
    encoder_TYPE = OneHotEncoder()
    encoder_TYPE = encoder_TYPE.fit(enc_df["TYPE"])
    encoded_data = encoder_TYPE.transform(df["TYPE"]).values
    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder_TYPE.get_feature_names_out()
    )
    df = pd.concat([df.drop(["TYPE"], axis=1), encoded_df], axis=1)

    # Scale the PRICE column
    scaler_PRICE = StandardScaler()
    scaler_PRICE = scaler_PRICE.fit(enc_df[["PRICE"]])
    df["PRICE_SCALED"] = scaler_PRICE.transform(df[["PRICE"]])
    df = df.drop(["PRICE"], axis=1)

    # Scale the BATH column
    scaler_BATH = StandardScaler()
    scaler_BATH = scaler_BATH.fit(enc_df[["BATH"]])
    df["BATH_SCALED"] = scaler_BATH.transform(df[["BATH"]])
    df = df.drop(["BATH"], axis=1)

    # Scale the PROPERTYSQFT column
    scaler_PROPERTYSQFT = StandardScaler()
    scaler_PROPERTYSQFT = scaler_PROPERTYSQFT.fit(enc_df[["PROPERTYSQFT"]])
    df["PROPERTYSQFT_SCALED"] = scaler_PROPERTYSQFT.transform(df[["PROPERTYSQFT"]])
    df = df.drop(["PROPERTYSQFT"], axis=1)

    # Scale the LATITUDE column
    scaler_LATITUDE = StandardScaler()
    scaler_LATITUDE = scaler_LATITUDE.fit(enc_df[["LATITUDE"]])
    df["LATITUDE_SCALED"] = scaler_LATITUDE.transform(df[["LATITUDE"]])
    df = df.drop(["LATITUDE"], axis=1)

    # Scale the LONGITUDE column
    scaler_LONGITUDE = StandardScaler()
    scaler_LONGITUDE = scaler_LONGITUDE.fit(enc_df[["LONGITUDE"]])
    df["LONGITUDE_SCALED"] = scaler_LONGITUDE.transform(df[["LONGITUDE"]])
    df = df.drop(["LONGITUDE"], axis=1)

    # Encoder map
    enc_map = dict()
    enc_map["TYPE"] = encoder_TYPE

    # Scaler map
    scaler_map = dict()
    scaler_map["PRICE"] = scaler_PRICE
    scaler_map["BATH"] = scaler_BATH
    scaler_map["PROPERTYSQFT"] = scaler_PROPERTYSQFT
    scaler_map["LATITUDE"] = scaler_LATITUDE
    scaler_map["LONGITUDE"] = scaler_LONGITUDE

    X = np.array(df.drop(["BEDS"], axis=1), dtype="float32")
    y = np.array(df[["BEDS"]], dtype="float32")

    # Reshape the data
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = y.reshape(y.shape[0], 1, y.shape[1])

    return X, y, enc_map, scaler_map


def load_test_data(enc_map, scaler_map):
    """
    Function to load the test data.
    Requires encoders and scalers used for the training data as arguments.
    """
    # Load the testing data
    test_data_path = "test_data.csv"
    df = pd.read_csv(test_data_path)

    # Drop some columns
    cols_to_drop = [
        "BROKERTITLE",
        "ADDRESS",
        "STATE",
        "MAIN_ADDRESS",
        "ADMINISTRATIVE_AREA_LEVEL_2",
        "LOCALITY",
        "SUBLOCALITY",
        "STREET_NAME",
        "LONG_NAME",
        "FORMATTED_ADDRESS",
    ]
    df = df.drop(cols_to_drop, axis=1)

    # Encode the TYPE column
    encoded_data = enc_map["TYPE"].transform(df["TYPE"]).values
    encoded_df = pd.DataFrame(
        encoded_data, columns=enc_map["TYPE"].get_feature_names_out()
    )
    df = pd.concat([df.drop(["TYPE"], axis=1), encoded_df], axis=1)

    # Scale the PRICE column
    df["PRICE_SCALED"] = scaler_map["PRICE"].transform(df[["PRICE"]])
    df = df.drop(["PRICE"], axis=1)

    # Scale the BATH column
    df["BATH_SCALED"] = scaler_map["BATH"].transform(df[["BATH"]])
    df = df.drop(["BATH"], axis=1)

    # Scale the PROPERTYSQFT column
    df["PROPERTYSQFT_SCALED"] = scaler_map["PROPERTYSQFT"].transform(
        df[["PROPERTYSQFT"]]
    )
    df = df.drop(["PROPERTYSQFT"], axis=1)

    # Scale the LATITUDE column
    df["LATITUDE_SCALED"] = scaler_map["LATITUDE"].transform(df[["LATITUDE"]])
    df = df.drop(["LATITUDE"], axis=1)

    # Scale the LONGITUDE column
    df["LONGITUDE_SCALED"] = scaler_map["LONGITUDE"].transform(df[["LONGITUDE"]])
    df = df.drop(["LONGITUDE"], axis=1)

    X = np.array(df, dtype="float32")

    # Reshape the data
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X


# Function to round model output to nearest output in training/testing data
def get_pred(pred):
    pred = np.array([ele[0][0] for ele in pred])
    target = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            20,
            24,
            30,
            32,
            35,
            36,
            40,
            42,
            50,
        ]
    )

    # Pre-allocate array for efficiency
    rounded_pred = np.zeros_like(pred)

    # Calculate absolute differences with broadcasting
    diffs = np.abs(target[:, np.newaxis] - pred)

    # Find indices of nearest integers using argmin
    min_indices = np.argmin(diffs, axis=0)

    # Assign nearest integers from int_list
    rounded_pred = target[min_indices]

    # Ensure output is int32
    return rounded_pred.astype(int).tolist()


# Function to write to output file
def write_out(pred):
    output = dict()
    output["BEDS"] = pred
    pd.DataFrame(output).to_csv("output.csv", index=False)


# Main function
def main():
    # Set random seed
    np.random.seed(963713)

    # Load the data
    X, y, enc_map, scaler_map = load_train_data()

    # Define the model
    model = NeuralNetwork()
    model.add(Dense(X.shape[-1], 128))
    model.add(Activation(leaky_relu, leaky_relu_prime))
    model.add(Dense(128, 64))
    model.add(Activation(leaky_relu, leaky_relu_prime))
    model.add(Dense(64, 32))
    model.add(Activation(leaky_relu, leaky_relu_prime))
    model.add(Dense(32, y.shape[-1]))
    model.add(Activation(relu, relu_prime))

    # Assign the loss function
    model.use(mape, mape_prime)

    # Train the model
    model, acc = model.fit(
        X, y, epochs=500, initial_learning_rate=0.001, decay_rate=0.95
    )

    # Perform prediction for the testing data
    X = load_test_data(enc_map, scaler_map)
    pred = model.predict(X)
    pred = get_pred(pred)

    # Write predictions to output file
    write_out(pred)


if __name__ == "__main__":
    main()
