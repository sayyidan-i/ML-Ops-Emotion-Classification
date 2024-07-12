import tensorflow as tf

LABEL_KEY = 'Emotion'
FEATURE_KEY = "Comment"

def transformed_name(key):
    #mengubah nama field yang telah di transform
    return key + '_xf'

def preprocessing_fn(inputs):
    """
    transform fitur agar lowercase dan label menjadi integer
    
    """
    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    

    
    return outputs
