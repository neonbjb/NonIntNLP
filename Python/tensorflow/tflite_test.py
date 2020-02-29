import tensorflow as tf
import numpy as np
from transformers import GPT2Tokenizer
print(tf.__version__)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

saved_model_dir = "c:/Users/jbetk/Documents/data/ml/saved_models/sentiment_mse_gp2_yelp_amazon/"
interpreter = tf.lite.Interpreter(model_path=saved_model_dir + "/converted_model.tflite")
print("Inputs: ", interpreter.get_input_details())
print("Outputs: ", interpreter.get_output_details())

phrases = [
    " This product ruined my day. Not figuratively, I stuck my fingers together and couldn't get them apart! I would never recommend this to anyone! Luckily I could return it..",
    # 5 star:
    " This product has been the standard for woodworking pros for decades. Before the 800lb gorilla swaggered into the room, taking the market by storm and strong-arming his way to happily humming registers all over the country, Titebond was THE glue to beat.",
    # 4 star:
    " good glue, but had hardened glue stuck in the spout and is clogged have to unscrew and use a popsicle stick to use, i will try and use isopropyl to dissolve it so i can use the spout, kinda annoying",
    # 3 star:
    " Strong as can be, but stain will discolor. Make sure you wipe it off completely before it dries.",
    # 2 star:
    " Did not use this glue immediately. About 4 weeks later, I could not get any glue to squeeze out thru the dispenser tip. I removed the cap and discovered lumps in the glue.",
]


def pad_zero(inputs, seq_len):
    for k in inputs:
        output = np.zeros(seq_len, dtype='int32')
        output[:len(inputs[k])] = np.asarray(inputs[k])
        inputs[k] = output
    return inputs


for phrase in phrases:
    enc = pad_zero(tokenizer.encode_plus(phrase, add_special_tokens=True, max_length=128), 128)
    phrase_encoded = []
    for k in enc.keys():
        phrase_encoded.append(enc[k])

    interpreter.reset_all_variables()
    interpreter.allocate_tensors()

    np.copyto(interpreter.tensor(interpreter.get_input_details()[0]["index"])(), phrase_encoded[0])
    interpreter.invoke()
    print(("inference [%f]: " + phrase) % (interpreter.tensor(interpreter.get_output_details()[0]["index"])()))

