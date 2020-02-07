from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import orjson
import numpy as np

from transformers import GPT2Tokenizer

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._responses = []

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def append_result(self, text, response):
        with self._condition:
            self._responses.append({'text': text, 'response': response})


def _create_rpc_callback(actual_text, actual_score, result_counter):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(result_future.result().outputs['final_linear'].float_val)
            result_counter.append_result(actual_text, response)
            print("%f/%f for {%s}" % (response, actual_score, actual_text))
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback

def load_data(filename):
    data = orjson.loads(open(filename, "rb").read())
    return np.asarray(data['input_id']), np.asarray(data['label'])

def do_inference():
    i = 0
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test_data_set, orig_stars = load_data("C:/Users/jbetk/Documents/data/ml/sentiment_analysis/outputs/gpt2/validation.json")

    channel = grpc.insecure_channel("192.168.56.101:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    num_tests = 150
    result_counter = _ResultCounter(num_tests, 50)
    for _ in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'gpt2'
        encoded_text = test_data_set[i]
        original_response = orig_stars[i]
        i += 1
        request.inputs["input_ids"].CopyFrom(
            tf.make_tensor_proto(encoded_text, shape=[1, encoded_text.size]))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(tokenizer.decode(encoded_text), original_response, result_counter))

if __name__ == '__main__':
    do_inference()