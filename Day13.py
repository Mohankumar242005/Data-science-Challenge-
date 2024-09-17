import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import matplotlib.pyplot as plt

def cnn_mnist_classification():
    
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) / 255.0

    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    
    history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (CNN): {test_acc}")

    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def rnn_imdb_classification():
    
    vocab_size = 10000
    max_len = 100

    (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=vocab_size)
    train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
    test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=max_len)

    
    model = models.Sequential([
        layers.Embedding(vocab_size, 32, input_length=max_len),
        layers.SimpleRNN(32, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))

    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"Test accuracy (RNN): {test_acc}")

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    print("Choose the model to run:")
    print("1. CNN (Image Classification - MNIST)")
    print("2. RNN (Text Classification - IMDb)")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        cnn_mnist_classification()
    elif choice == '2':
        rnn_imdb_classification()
    else:
        print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()
