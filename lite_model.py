import tensorflow as tf
model = tf.keras.models.load_model('./models/best_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()
with open('models/pneumonia_model.tflite', 'wb') as f:
    f.write(lite_model)