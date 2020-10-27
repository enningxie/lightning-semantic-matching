# -*- coding: utf-8 -*-
# Created by xieenning at 2020/10/26
import os
from tqdm import trange
from time import time
from contextlib import contextmanager
import torch
import statistics
import numpy as np
from torch.onnx import export
from typing import Union
from src.model import SemanticMatchingClassifier
from transformers import PreTrainedModel, PreTrainedTokenizer, BertTokenizerFast, ElectraTokenizer
from transformers.file_utils import ModelOutput
from onnxruntime_tools import optimizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


# TODO: refactor
def onnx_batch_generator(features, batch_size=128):
    input_ids = features["input_ids"].numpy()
    token_type_ids = features["token_type_ids"].numpy()
    attention_mask = features["attention_mask"].numpy()

    if len(input_ids) % batch_size == 0:
        n = len(input_ids) // batch_size
    else:
        n = len(input_ids) // batch_size + 1

    for j in range(n):
        yield {
            "input_ids": input_ids[j * batch_size:(j + 1) * batch_size],
            "token_type_ids": token_type_ids[j * batch_size:(j + 1) * batch_size],
            "attention_mask": attention_mask[j * batch_size:(j + 1) * batch_size]}


def calculate_inference_time(model: Union[PreTrainedModel, InferenceSession], inputs):
    # Keep track of the inference time
    time_buffer = []
    if isinstance(model, InferenceSession):
        inputs = {k: to_numpy(v) for k, v in inputs.items()}

    # Warm up the model
    # transformers model warming up.
    model_output = None
    for _ in trange(10, desc="Warming up."):
        if isinstance(model, InferenceSession):
            model_output = model.run(None, inputs)
        else:
            model_output = model(**inputs)
    model_output_ = model_output[0]

    # Compute
    for _ in trange(100, desc=f"Tracking inference time."):
        with track_infer_time(time_buffer):
            if isinstance(model, InferenceSession):
                model.run(None, inputs)
            else:
                model(**inputs)

    # Store the result
    return model_output_, time_buffer


def infer_shapes(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model.
    :return:
    """

    tokens = tokenizer("This is a sample sentence1.", "This is a sample sentence2", padding=True,
                       truncation="longest_first", max_length=64, return_tensors='pt')
    outputs = model(**tokens)
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Generate input names & axes
    input_names = list(tokens.keys())
    input_dynamic_axes = {k: {0: 'batch', 1: 'sequence'} for k, v in tokens.items()}

    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    # Generate output names & axes
    output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: {0: 'batch'} for k in output_names}

    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_names, output_names, dynamic_axes, tokens


def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any None
    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    print("Ensuring inputs are in correct order")

    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # start at index 1 to skip "self" argument
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f"{arg_name} is not present in the generated input list.")
            break

    print("Generated inputs order: {}".format(ordered_input_names))
    return ordered_input_names, tuple(model_args)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_to_onnx(model: PreTrainedModel, output_path, opset: int = 12):
    onnx_output_path = os.path.join(output_path, 'checkpoint_without_optimize.onnx')
    onnx_optimized_output_path = os.path.join(output_path, 'checkpoint_with_optimize.onnx')
    onnx_optimized_fp16_output_path = os.path.join(output_path, 'checkpoint_with_optimize_fp16.onnx')

    model.eval()
    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(tmp_model, tmp_tokenizer)
        ordered_input_names, model_args = ensure_valid_input(model, tokens, input_names)
        print(f"Model input names: {ordered_input_names}.")
        export(model, model_args, onnx_output_path,
               input_names=ordered_input_names,
               output_names=output_names,
               dynamic_axes=dynamic_axes,
               verbose=True, opset_version=opset)
        print(f"Finished output checkpoint_without_optimize.onnx to {output_path}.")

    optimized_model = optimizer.optimize_model(onnx_output_path, model_type='bert', num_heads=12, hidden_size=768,
                                               use_gpu=True)
    optimized_model.save_model_to_file(onnx_optimized_output_path)
    print(f"Finished output checkpoint_with_optimize.onnx to {output_path}.")
    optimized_model.convert_model_float32_to_float16()
    optimized_model.save_model_to_file(onnx_optimized_fp16_output_path)
    print(f"Finished output checkpoint_with_optimize_fp16.onnx to {output_path}.")


def onnx_runtime_inference(onnx_model_path, tokenizer, sentence1_list, sentence2_list, batch_size=None):
    # load onnx_model
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    # options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CUDA backend
    ort_session = InferenceSession(onnx_model_path, options,
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    tokens = tokenizer(sentence1_list, sentence2_list, padding=True,
                       truncation="longest_first", max_length=64, return_tensors='pt')

    if batch_size is None:
        ort_inputs = {k: to_numpy(v) for k, v in tokens.items()}
        ort_outs = ort_session.run(None, ort_inputs)[0]
    else:
        batches = onnx_batch_generator(tokens, batch_size=batch_size)
        ort_outs = np.vstack([ort_session.run(None, b)[0] for b in batches])
    return ort_outs


if __name__ == '__main__':
    experiment_folder = '/Data/enningxie/Codes/lightning-semantic-matching/src/experiments/version_26-10-2020--10-16-16'
    tmp_checkpoint_path = os.path.join(experiment_folder, 'checkpoints.ckpt')
    # /Data/public/pretrained_models/chinese-electra-180g-base-discriminator
    # /Data/public/pretrained_models/pytorch/chinese-bert-wwm-ext
    tmp_model_name_or_path = "/Data/public/pretrained_models/chinese-electra-180g-large-discriminator"
    tmp_tokenizer = ElectraTokenizer.from_pretrained(tmp_model_name_or_path)
    tmp_model = SemanticMatchingClassifier.load_from_checkpoint(checkpoint_path=tmp_checkpoint_path)

    # 1
    # transformers model convert to onnx model
    convert_to_onnx(tmp_model, experiment_folder)

    # 2
    # inference example.
    onnx_without_optimize_model_path = os.path.join(experiment_folder, 'checkpoint_without_optimize.onnx')
    onnx_with_optimize_model_path = os.path.join(experiment_folder, 'checkpoint_with_optimize.onnx')
    onnx_with_optimize_fp16_model_path = os.path.join(experiment_folder, 'checkpoint_with_optimize_fp16.onnx')
    tmp_sentence1_list = ['我不打算买车', '好的', '不好']
    tmp_sentence2_list = ['买好了', '知道', '不用了']
    transformers_inputs = tmp_tokenizer(tmp_sentence1_list, tmp_sentence2_list, padding=True,
                                        truncation="longest_first", max_length=64, return_tensors='pt')
    # transformers model inference
    tmp_model.eval()
    with torch.no_grad():
        transformers_output = tmp_model(**transformers_inputs)
    print(f"transformers model output: {transformers_output}.")

    onnx_without_optimize_output = onnx_runtime_inference(onnx_without_optimize_model_path, tmp_tokenizer,
                                                          tmp_sentence1_list, tmp_sentence2_list)
    print(f"ONNX without optimize output: {onnx_without_optimize_output}.")
    onnx_with_optimize_output = onnx_runtime_inference(onnx_with_optimize_model_path, tmp_tokenizer, tmp_sentence1_list,
                                                       tmp_sentence2_list)
    print(f"ONNX with optimize output: {onnx_with_optimize_output}.")
    onnx_with_optimize_fp16_output = onnx_runtime_inference(onnx_with_optimize_fp16_model_path, tmp_tokenizer,
                                                            tmp_sentence1_list, tmp_sentence2_list)
    print(f"ONNX with optimize fp16 output: {onnx_with_optimize_fp16_output}.")

    # 3
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # 4
    # ## check op
    # # Load the ONNX model
    # model = onnx.load(onnx_file_path)
    #
    # # Check that the model is well formed
    # onnx.checker.check_model(model)

    # 5
    # transformers_inputs_ = tmp_tokenizer(["我买好了"]*100, ["早就买了"]*100, padding="max_length",
    #                                      truncation="longest_first", max_length=64, return_tensors='pt')
    # transformers_inputs_ = {k: v.to('cuda:0') for k, v in transformers_inputs_.items()}
    # print(f"Model device: {tmp_model.device}.")
    # tmp_model.to('cuda:0')
    # tmp_model.eval()
    # with torch.no_grad():
    #     model_output_, time_buffer = calculate_inference_time(tmp_model, transformers_inputs_)
    #
    # print(f"Average latency: {statistics.mean(time_buffer)*1000:.2f}ms")
    #
    # options = SessionOptions()
    # # options.intra_op_num_threads = 1
    # options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    #
    # # Load the model as a graph and prepare the CUDA backend
    # ort_session = InferenceSession(onnx_with_optimize_fp16_model_path, options,
    #                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # _, time_buffer = calculate_inference_time(ort_session, transformers_inputs_)
    # print(f"Average latency: {statistics.mean(time_buffer) * 1000:.2f}ms")
    # print('Break point.')
