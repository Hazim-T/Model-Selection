from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def show_images(images, labels, class_names, num_images=32):
    plt.figure(figsize=(12, 12))
    for i in range(min(num_images, len(images))):
        plt.subplot(6, 6, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        true_label = class_names[labels[i].argmax()]
        plt.title(true_label)
    plt.tight_layout()
    plt.show()


def plot_predictions(generator, model, class_names):
    images, labels = next(generator)
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(15, 15))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

        true_label = class_names[true_classes[i]]
        predicted_label = class_names[predicted_classes[i]]

        color = 'blue' if predicted_classes[i] == true_classes[i] else 'red'

        # Display the predicted label in the chosen color
        plt.title(predicted_label, color=color)

    plt.tight_layout()
    plt.show()


def limited_image_generator(generator, limit):
    images_read = 0
    while images_read < limit:
        batch_x, batch_y = next(generator)
        images_read += batch_x.shape[0]

        # Yield only up to the limit in the last batch
        if images_read > limit:
            yield batch_x[:limit - (images_read - batch_x.shape[0])], batch_y[:limit - (images_read - batch_x.shape[0])]
        else:
            yield batch_x, batch_y


def concatenate_histories(history_pre_tuning, history_post_tuning):
    combined_history = {
        'accuracy': history_pre_tuning.history['accuracy'] + history_post_tuning.history['accuracy'],
        'val_accuracy': history_pre_tuning.history['val_accuracy'] + history_post_tuning.history['val_accuracy'],
        'loss': history_pre_tuning.history['loss'] + history_post_tuning.history['loss'],
        'val_loss': history_pre_tuning.history['val_loss'] + history_post_tuning.history['val_loss'],
    }
    return combined_history


def plot_accuracy(histories_dict):
    plt.figure(figsize=(12, 8))

    for model_name, history in histories_dict.items():
        plt.plot(history['accuracy'], label=f'{model_name} Training Accuracy')
        plt.plot(history['val_accuracy'], linestyle='--', label=f'{model_name} Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(histories_dict):
    plt.figure(figsize=(12, 8))

    for model_name, history in histories_dict.items():
        plt.plot(history['loss'], label=f'{model_name} Training Loss')
        plt.plot(history['val_loss'], linestyle='--', label=f'{model_name} Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_final_metrics(histories_dict):
    final_metrics = {
        "Model": list(histories_dict.keys()),
        "Final Training Accuracy": [history['accuracy'][-1] for history in histories_dict.values()],
        "Final Validation Accuracy": [history['val_accuracy'][-1] for history in histories_dict.values()],
        "Final Training Loss": [history['loss'][-1] for history in histories_dict.values()],
        "Final Validation Loss": [history['val_loss'][-1] for history in histories_dict.values()]
    }

    summary_df = pd.DataFrame(final_metrics)
    print(summary_df)