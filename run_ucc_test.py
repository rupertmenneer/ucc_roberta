from ucc_classifier_pkg.run_classifier import  run_classifer

if __name__ == '__main__':
    test_data_to_verify_classifier = run_classifer(128)
    print(f"classifier ROC AUC on test data {test_data_to_verify_classifier.results}")