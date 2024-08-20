import torch
import psutil
import time
from prettytable import PrettyTable
import datetime
import os
import numpy as np
import onnxruntime as ort
import json


#helper functions
def contains_matmul(string):
    return "MatMul" in string


def generate_table(profile_json):


# Path to your JSON file
  file_path = profile_json
  table = PrettyTable()
  table.field_names = ["Index","Layer Name" ,"Op", "Time (ms)", "Input Shape", "Output Shape","Backend"]
  session_table=PrettyTable()
  session_table.field_names = ["Name ", "Time (ms)"]
  # Open and load the JSON file
  with open(file_path, 'r') as file:
      data = json.load(file)
  os.remove(profile_json)
  # Iterate through each element/dictionary
  time=0
  counter=1

  for element in data:

      if element['cat']=='Node' and element['name'].endswith('time'):
          # print(element)
          o_shape=element['args']['output_type_shape'][0]['float']
          i_shape=element['args']['input_type_shape'][0]['float']
          duration=element['dur']
          layer=element['name']
          provider=element['args']['provider']
          op=element['args']['op_name']


          if contains_matmul(op):
            i_shape=[element['args']['input_type_shape'][0]['float'],element['args']['input_type_shape'][1]['float']] if len(element['args']['input_type_shape'])>1 else element['args']['input_type_shape'][0]['float']


          table.add_row([counter,layer,op,float(duration/1000),i_shape,o_shape,provider])
          counter+=1
          time+=duration
      if element['cat']=='Session':
        duration=float(element['dur']/1000)
        session_table.add_row([element['name'],duration])




  return table,session_table





class ModelProfiler:
    def __init__(self, model=None, use_cuda=False,type='torch',providers=None,intra_op_num_threads=None,input_data=None,export_txt=False,Sort_layers=True):
      self.type=type
      if type=='torch':
        self.model = model.cpu().eval()
        self.use_cuda = use_cuda
        self.type=type

        if use_cuda:
            self.model = self.model.cuda()

        self.layer_types = {}
        self.layer_times = {}
        self.layer_cpu_memory = {}
        self.layer_gpu_memory = {}
        self.layer_input_shapes = {}
        self.layer_output_shapes = {}
        self.hooks = []
        self.total_time = 0
        self.max_memory = 0
        self.start_time = None

      if type=='onnx':
        self.providers=providers
        if providers is None:
          self.providers = ['CPUExecutionProvider']
        if intra_op_num_threads is None:
          self.intra_op_num_threads=os.cpu_count()

        self.model_path=model
        self.session_options = ort.SessionOptions()
        if self.model_path.endswith('.ort'):
          self.session_options
        self.session_options.enable_profiling = True
        self.session_options.intra_op_num_threads = os.cpu_count()
        self.session = ort.InferenceSession(self.model_path,providers=self.providers,sess_options=self.session_options)

      self.export_txt=export_txt
      self.Sort_layers=Sort_layers




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
        self.hooks = []

    def pre_forward_hook(self, name):
        def hook(layer, input):
            self.start_time = time.time()
            process = psutil.Process()
            current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
            self.layer_cpu_memory[name] = current_cpu_memory
            if isinstance(input, torch.Tensor):
              self.layer_input_shapes[name] = input.shape
            else:
              for inp in input:
                if isinstance(inp, torch.Tensor):
                  self.layer_input_shapes[name] = inp.shape
                  break
                else:
                  pass

            if self.use_cuda:
                torch.cuda.synchronize()
                self.layer_gpu_memory[name] = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        return hook

    def post_forward_hook(self, name):
        def hook(layer, input, output):
            end_time = time.time()
            process = psutil.Process()
            current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

            elapsed_time = end_time - self.start_time

            self.layer_times[name] = elapsed_time
            self.layer_cpu_memory[name] = current_cpu_memory
            # self.layer_output_shapes[name] = output.shape if isinstance(output, torch.Tensor) else [out.shape for out in output]
            if isinstance(output, torch.Tensor):
              self.layer_output_shapes[name] = output.shape
            else:
              for out in output:
                # print("here out ", out)
                if isinstance(out, torch.Tensor):
                  self.layer_output_shapes[name] = out.shape
                  break
                else:
                  pass
            if self.use_cuda:
                torch.cuda.synchronize()
                self.layer_gpu_memory[name] = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
            self.max_memory = max(self.max_memory, current_cpu_memory)
        return hook

    def profile(self, input_data=None):
      if self.type=='torch':
        if self.use_cuda:
            input_data = input_data.cuda()
        self.register_hooks()
        start_total_time = time.time()
        self.model(input_data)
        self.total_time = time.time() - start_total_time
        self.remove_hooks()
      if self.type=='onnx':
        if input_data is None:
          input_details = self.session.get_inputs()
          input_shape = input_details[0].shape
          input_data=np.random.rand(*input_shape).astype(np.float32)
        self.session.run(None, {input_details[0].name: input_data})
        self.profile_file=self.session.end_profiling()


    def print_profiling_info(self, print_io_shape=True):
      if self.type=='torch':
        table = PrettyTable()
        field_names = ["Row ID", "Layer", "Type", "Time (s)", "CPU Memory (MB)"]
        if self.use_cuda:
            field_names.append("GPU Memory (MB)")
        if print_io_shape:
            field_names.extend(["Input Shape", "Output Shape"])
        table.field_names = field_names

        for i, layer in enumerate(self.layer_times):
            row = [i + 1, layer, self.layer_types[layer], f"{self.layer_times[layer]:.6f}", f"{self.layer_cpu_memory[layer]:.2f}"]
            if self.use_cuda:
                row.append(f"{self.layer_gpu_memory.get(layer, 0):.2f}")
            if print_io_shape:
                o_layers=list(self.layer_output_shapes.keys())
                i_layers=list(self.layer_input_shapes.keys())
                if layer not in o_layers:
                  self.layer_output_shapes[layer]=[]
                if layer not in i_layers:
                  self.layer_input_shapes[layer]=[]
                row.extend([self.layer_input_shapes[layer], self.layer_output_shapes[layer]])
            table.add_row(row)

        print("Layer Profiling Information:")

        if self.Sort_layers:
          table.sortby="Time (s)"
          table.reversesort=True
          print(table)
        if self.export_txt:
          file_name=f'torch_profile__{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
          with open(file_name, 'w') as w:
            w.write(str(table))

        print(f"\nTotal Inference Time: {self.total_time:.6f} seconds")
        print(f"Max Memory Consumption: {self.max_memory:.2f} MB")
        self.table=table

      if self.type=='onnx':
        table,session_table=generate_table(self.profile_file)
        if print_io_shape==False:
          table.del_column('Input Shape')
          table.del_column('Output Shape')
        if self.Sort_layers:
          table.sortby="Time (ms)"
          table.reversesort=True
        print(table)
        self.table=table
        if self.export_txt:
          file_name=f'onnxruntime_profile__{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
          with open(file_name, 'w') as w:
            w.write(str(table))
        print("Session info ")
        print(session_table)





    def print_top_k_layers(self, k, print_io_shape=True):
      if self.type=='torch':

        sorted_layers = sorted(self.layer_times.items(), key=lambda x: x[1], reverse=True)
        table = PrettyTable()
        field_names = ["Row ID", "Layer", "Type", "Time (s)"]
        if print_io_shape:
            field_names.extend(["Input Shape", "Output Shape"])
        table.field_names = field_names

        for i, (layer, time_taken) in enumerate(sorted_layers[:k]):
            row_id = list(self.layer_times.keys()).index(layer) + 1
            row = [row_id, layer, self.layer_types[layer], f"{time_taken:.6f}"]
            if print_io_shape:
                row.extend([self.layer_input_shapes[layer], self.layer_output_shapes[layer]])
            table.add_row(row)

        print(f"Top {k} Layers by Execution Time:")
        print(table)
      if self.type=='onnx':
        self.sorted_table=PrettyTable()
        headers=self.table.field_names
        self.sorted_table.field_names=headers
        rows=self.table.rows
        rows.sort(key=lambda x: x[3], reverse=True)
        for row in rows[:k]:
          self.sorted_table.add_row(row)


        if print_io_shape==False:
          self.sorted_table.del_column('Input Shape')
          self.sorted_table.del_column('Output Shape')


        print(f"Top {k} Layers by Execution Time:")
        print(self.sorted_table)








