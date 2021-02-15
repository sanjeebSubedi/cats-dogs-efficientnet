# cats-dogs-efficientnet
### New Features feb 05
- New hyperparameter named `load_pretrained_weights` to choose whether to load pretrained CNN or not.
- Feature to save checkpoint after a certain number of epochs set by the hyperparameter `checkpoint_save_frequency`.
- Loads a saved checkpoint if `load_checkpoint` is set to true.

This project uses efficientnet to classify pictures of cats and dogs in the kaggle cats vs dogs dataset.
