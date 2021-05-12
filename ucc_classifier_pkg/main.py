
# Press the green button in the gutter to run the script.
import numpy as np
from ucc_classifier_pkg.ucc_classifer_wrapper import ClassifierWrapper
import pandas as pd
from sklearn import metrics

if __name__ == '__main__':
    config = {
        'model_name': 'roberta-base',
        'attributes': ['antagonise', 'condescending', 'dismissive', 'generalisation',
                       'generalisation_unfair', 'unhealthy', 'hostile', 'sarcastic'],
        'n_classes': 8,
        'n_gpus': 0,
        'batch_size': 32,
        'max_token_len': 128,
        'checkpoint_dir': "/Users/rupert/Documents/ucc_classifier/assets/roberta-ucc-model-v1.0.0.ckpt"
    }
    classifier = ClassifierWrapper(config)
    # data
    test = pd.read_csv('../assets/test.csv')
    # Make 'unhealthy' class (so all postive classes are minority class)
    test['unhealthy'] = np.where(test['healthy'] == 1, 0, 1)
    test_comment_list = list(test['comment'].astype(str).sample(64, random_state=8))
    # get model predictions
    test_predictions = classifier.classify_raw_comments(test_comment_list, 'bool')
    print(test_predictions[:,0].shape)
    print(test_predictions[:, 0])
    # get true labels
    test_labels = test[config['attributes']].sample(64, random_state=8)
    test_labels = np.array(test_labels.values)
    # print(test_labels.shape, test_predictions.shape)
    # print(metrics.roc_auc_score(test_labels, test_predictions))