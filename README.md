# cats-dogs-efficientnet
### Updates on feb 06
- Custom neural Network class `efficientNetCustom` is now used to load efficientnet model instead of the `load_model` function.
- Added option to train the last layer only with the `train_only_last_layer` hyperparameter.
- Training now shows the duration of each epoch in seconds.
- Loss is now rounded to 5 decimal places like accuracy.


### New Features feb 05
- New hyperparameter named `load_pretrained_weights` to choose whether to load pretrained (Imagenet) CNN or not.
- Feature to save checkpoint after a certain number of epochs set by the hyperparameter `checkpoint_save_frequency`.
- Loads a saved checkpoint if `load_checkpoint` is set to true.

-------------------------------

#### This is a beginner's project which uses efficientnet to classify pictures of cats and dogs in the kaggle cats vs dogs dataset.
