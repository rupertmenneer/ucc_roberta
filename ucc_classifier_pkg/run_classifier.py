import numpy as np
from ucc_classifier_pkg.ucc_classifer_wrapper import ClassifierWrapper
import pandas as pd
from sklearn import metrics


class run_classifer:

    def __init__(self, sample_size):
        self.results = self.run_test(sample_size)

    @staticmethod
    def run_test(sample_size=128):
        config = {
            'model_name': 'roberta-base',
            'attributes': ['antagonise', 'condescending', 'dismissive', 'generalisation',
                           'generalisation_unfair', 'unhealthy', 'hostile', 'sarcastic'],
            'n_classes': 8,
            'n_gpus': 0,
            'batch_size': 32,
            'max_token_len': 128,
            'checkpoint_dir': "/Users/rupert/Documents/GitHub/ucc_roberta/assets/roberta-ucc-model-v1.0.0.ckpt"
        }
        classifier = ClassifierWrapper(config)
        # data
        test = pd.read_csv('/Users/rupert/Documents/GitHub/ucc_roberta/assets/test.csv')
        # Make 'unhealthy' class (so all postive classes are minority class)
        test['unhealthy'] = np.where(test['healthy'] == 1, 0, 1)
        test_comment_list = list(test['comment'].astype(str).sample(sample_size, random_state=8))
        # get model predictions
        test_predictions = classifier.classify_raw_comments(test_comment_list, 'float')
        # get true labels
        test_labels = test[config['attributes']].sample(sample_size, random_state=8)
        test_labels = np.array(test_labels.values)
        auc_scores = {}
        for i, attribute in enumerate(config['attributes']):
            auc_scores[attribute] = metrics.roc_auc_score(test_labels[:, i].astype(int), test_predictions[:, i])
        auc_scores['ALL'] = metrics.roc_auc_score(test_labels, test_predictions)
        return auc_scores
