# utils/profiler.py

import torch
import torchvision.models.segmentation as models
from prettytable import PrettyTable
import time
from collections import OrderedDict
import psutil

class ModelProfiler:
    def __init__(self, model):
        self.model = model
        self.layer_types = {}
        self.layer_times = {}
        self.layer_memory = {}
        self.hooks = []
        self.total_time = 0

    def register_hooks(self):
        for name, layer in self.model.named_modules():
            self.layer_types[name] = layer.__class__.__name__
            pre_hook = layer.register_forward_pre_hook(self.pre_forward_hook(name))
            post_hook = layer.register_forward_hook(self.post_forward_hook(name))
            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def pre_forward_hook(self, name):
        def hook(layer, input):
            self.start_time = time.time()
            process = psutil.Process()
            self.layer_memory[name] = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        return hook

    def post_forward_hook(self, name):
        def hook(layer, input, output):
            end_time = time.time()
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

            elapsed_time = end_time - self.start_time

            self.layer_times[name] = elapsed_time
            self.layer_memory[name] = current_memory
        return hook

    def profile(self, input_data):
        self.register_hooks()
        start_total_time = time.time()
        self.model(input_data)
        self.total_time = time.time() - start_total_time
        self.remove_hooks()

    def print_profiling_info(self):
        table = PrettyTable()
        table.field_names = ["Row ID", "Layer", "Type", "Time (s)", "Current occupied CPU Memory (MB)"]

        for i, layer in enumerate(self.layer_times):
            time_taken = self.layer_times[layer]
            memory_used = self.layer_memory[layer]
            layer_type = self.layer_types[layer]
            table.add_row([i + 1, layer, layer_type, f"{time_taken:.6f}", f"{memory_used:.2f}"])

        print("Layer Profiling Information:")
        print(table)
        print(f"\nTotal Inference Time: {self.total_time:.6f} seconds")

    def print_top_k_layers(self, k):
        sorted_layers = sorted(self.layer_times.items(), key=lambda x: x[1], reverse=True)
        table = PrettyTable()
        table.field_names = ["Row ID", "Layer", "Type", "Time (s)"]

        for i, (layer, time_taken) in enumerate(sorted_layers[:k]):
            layer_type = self.layer_types[layer]
            row_id = list(self.layer_times.keys()).index(layer) + 1
            table.add_row([row_id, layer, layer_type, f"{time_taken:.6f}"])

        print(f"Top {k} Layers by Execution Time:")
        print(table)

