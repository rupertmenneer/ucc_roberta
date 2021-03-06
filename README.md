# ucc_roberta

The unhealthy comment corpus contains ~45,000 labeled comments. The labels consist of subtle sentiment attributes such as 'sarcasm'. The primary attribute is 'unhealthy'. This repo presents a RoBERTa model fine-tuned on the UCC task. It presents state-of-the-art results on this task with an average AUC ROC of ~0.84 compared to the AUC ROC of ~74 for the baseline model.

This repo presents the model in a form ready to make predictions on unseen data and is set up as a stand-alone service with JSON input and output. Unfortunately, the checkpoint file for this model is too large to be uploaded directly into repo, this will be required for correct predictions.


## model training

The UCC RoBERTa model was trained on Google Colab using a GPU. It was set up with Pytorch Lightning! ⚡️ it uses the Hugging Face library 🤗 to acquire RoBERTa. Hyper-parameter optimisation is achieved with the Ray Tune Library and the ASHA algorithm. Find this training process under model_training/

## statisical significance

The produced UCC RoBERTa model was evaluated and the improvement over the baseline model underwent McNemar statisical test. The improvement was found to be significant with a P-value of <0.0001.

## system requirements

the system will need all libraries versioned correctly as per either the requirements.txt file or the pip.lock file. Both have been provided for convenience.

## running as a service

this system is set up to run as a stand-alone service. Best practice is to run the main.py file from the command line via Gunicorn. It can also be run directly from the main.py file in an IDE for testing purposes.


