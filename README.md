# ucc_roberta

The unhealthy comment corpus contains ~45,000 labeled comments. The labels consist of subtle sentiment attributes such as 'sarcasm'. The primary attribute is 'unhealthy'. This repo presents a RoBERTa model fine-tuned on the UCC task. It presents state-of-the-art results on this task.

Please note: this repo presents the model in a form ready to make predictions on unseen data and is set up as a stand-alone service with JSON input and output. The checkpoint file for this model is too large to the repo and this will be required for correct predictions.

For the training process please reference the notebook under model_training.
