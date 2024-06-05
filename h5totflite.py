import tensorflow as tf
from keras_retinanet.models import load_model
from keras.layers import Input
from keras.models import Model

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size
    
def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

def convert_model_to_tflite(model_path = "model.h5", filename = "converted_model.tflite"):
  model = load_model(model_path)
  fixed_input = Input((416,416,3))
  fixed_model = Model(fixed_input,model(fixed_input))
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  tflite_model = converter.convert()
  open(filename, "wb").write(tflite_model)
  print(convert_bytes(get_file_size("converted_model.tflite"), "MB"))


  