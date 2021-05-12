from ucc_classifier_pkg import ucc_classifer_wrapper, run_classifier
import pandas as pd

if __name__ == '__main__':
    test_len = len(pd.read_csv('/Users/rupert/Documents/ucc_classifier/assets/test.csv'))
    test_data_to_verify_classifier = run_classifier.run_classifer(test_len)
    print(f"classifier ROC AUC on test data {test_data_to_verify_classifier.results}")
    fyp_data = pd.read_csv('/Users/rupert/Downloads/fyp-5g-twitter_fyp-5g-english-kw-match.csv')
    print(fyp_data.iloc[0]['twitter.tweet/text'])
    ucc_model = ucc_classifer_wrapper.ClassifierWrapper()
    fyp_classifier = ucc_model.classify_raw_comments(list(fyp_data['twitter.tweet/text']))
    attributes = ['antagonise', 'condescending', 'dismissive', 'generalisation',
                   'generalisation_unfair', 'unhealthy', 'hostile', 'sarcastic']
    fyp_classifier_pd = pd.DataFrame(fyp_classifier, columns=attributes)
    fyp_classifier_pd.to_csv('/Users/rupertw/Downloads/fyp_analysis-5g-v1.csv')