from transformers import BertConfig
from transformers import TFBertForMaskedLM
import data_preprocess
import tensorflow as tf

tf_train_dataset, tf_validation_dataset, tf_test_dataset = data_preprocess.preprocess(use_small_dataset=True)

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()
print('BERT config:', configuration)

# Initializing a model from the bert-base-uncased style configuration
teacher = TFBertForMaskedLM(configuration)

teacher.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[])

history = teacher.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=1
)
