#!/usr/bin/env python3
import torch
import torch.nn as nn
# import torch_mlu
# import torch_mlu.core.mlu_model as ct
# import torch_mlu.core.mlu_quantize as mlu_quantize
import os
import onnx
from torchinfo import summary
from queue import Queue
from onnx import shape_inference
import onnxruntime
import onnx.numpy_helper
from Operator import *
# ct.set_core_number(1)
# ct.set_core_version('MLU270')
torch.set_grad_enabled(False)


class Model(nn.Module):
    def __init__(self, onnx_model_path='export_ConvSigMul.onnx') -> None:
        super(Model, self).__init__()

        self.input: str = 'defualt_input'   # model 的输入
        self.output: str = 'defualt_output'  # model 的输出
        self.nodes_sorted = []              # 经过拓扑排序后的 node
        self.node_type: dict[str, str] = {}  # node 的类型
        self.layers_dict = nn.ModuleDict()  # 模型结构

        self.tensor_next: dict[str, set] = {}  # tensor 的前置 node

        self.node_pre: dict[str, list] = {}  # node 的前置(输入) tensor
        self.node_next: dict[str, list] = {}  # node 的后置(输出) tensor

        onnx_mode = self.__loadOnnx(onnx_model_path)    # 加载模型

        self.__parseOnnxModel(onnx_mode)  # 解析模型 并构建网络

    def forward(self, x):
        tensor_value: dict[str, torch.Tensor] = {}  # tensor_name: tensor
        tensor_value[self.input] = x
        tensor_count: dict[str, int] = {}  # 记录tensor的使用次数，若tensor_count[x]>=len(self.tensor_next) 则手动 del tensor

        for name in self.nodes_sorted:
            print('run:', name)
            node_type = self.node_type[name]
            ready_to_realse_tensor = []
            # 单输入单(多)输出
            if (node_type == 'Conv' or node_type == 'Sigmoid' or node_type == 'Shape' or node_type == 'Reshape' or node_type == 'Transpose' or node_type == 'Split' or node_type == 'Pow' or node_type == 'Slice' or node_type == 'Unsqueeze' or node_type == 'Cast' or node_type == 'MaxPool'):
                input = self.node_pre[name][0]
                x = self.layers_dict[name](tensor_value[input])
                tensor_count[input] = 1 if input not in tensor_count else tensor_count[input] + 1
                if tensor_count[input] >= len(self.tensor_next[input]):
                    ready_to_realse_tensor.append(input)
            elif (node_type == 'Mul' or node_type == 'Add'):
                # 两输入单输出
                input_0 = list(self.node_pre[name])[0]
                input_1 = list(self.node_pre[name])[1]
                if 'Constant' in input_1:  # 此时是 tensor * const
                    x = self.layers_dict[name](
                        tensor_value[input_0])
                else:  # 此时是 tensor * tensor
                    x = self.layers_dict[name](
                        tensor_value[input_0], tensor_value[input_1])

                tensor_count[input_0] = 1 if input_0 not in tensor_count else tensor_count[input_0] + 1
                if tensor_count[input_0] >= len(self.tensor_next[input_0]):
                    ready_to_realse_tensor.append(input_0)

                if 'Constant' not in input_1:
                    tensor_count[input_1] = 1 if input_1 not in tensor_count else tensor_count[input_1] + 1
                    if tensor_count[input_1] >= len(self.tensor_next[input_1]):
                        ready_to_realse_tensor.append(input_1)

            elif (node_type == 'Floor'):
                # 无输入无输出
                x = self.layers_dict[name]()
            elif (node_type == 'Resize'):
                # 两输入单输出
                input_x = self.node_pre[name][0]
                input_size = self.node_pre[name][3]
                x = self.layers_dict[name](
                    tensor_value[input_x], tensor_value[input_size])

                tensor_count[input_x] = 1 if input_x not in tensor_count else tensor_count[input_x] + 1
                if tensor_count[input_x] >= len(self.tensor_next[input_x]):
                    ready_to_realse_tensor.append(input_x)

                tensor_count[input_size] = 1 if input_size not in tensor_count else tensor_count[input_size] + 1
                if tensor_count[input_size] >= len(self.tensor_next[input_size]):
                    ready_to_realse_tensor.append(input_size)

            elif (node_type == 'Concat'):
                # 多输入单输出
                input_x = self.node_pre[name]
                input_x_tensor = []
                for input in input_x:
                    input_x_tensor.append(tensor_value[input])

                x = self.layers_dict[name](
                    input_x_tensor)

                for input in input_x:
                    tensor_count[input] = 1 if input not in tensor_count else tensor_count[input] + 1
                    if tensor_count[input] >= len(self.tensor_next[input]):
                        ready_to_realse_tensor.append(input)

            else:
                print("ERROR: operator \'", node_type, "\' not be implemented")
                os._exit(-1)
            # 释放已经用过的 tensor
            for released_tensor in ready_to_realse_tensor:
                del tensor_value[released_tensor]

            if (type(x) != torch.Tensor):
                # 算子输出为多个 tensor
                for idx, output_name in enumerate(self.node_next[name]):
                    tensor_value[output_name] = x[idx]
                    print('out:', x[idx].shape)
            else:
                # 算子输出为单个 tensor
                print('out:', x.shape)
                for output_name in self.node_next[name]:
                    tensor_value[output_name] = x
        return x

    def __loadOnnx(self, onnx_model_path) -> onnx.ModelProto:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        # 需要执行 shape_inference.infer_shape 推断模型各节点的输入、输出维度
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(onnx_model)
        return onnx_model

    def __getAttr(self, node, target: list[str]) -> dict[str, any]:
        '''
        用于获取 node 的 attributes
        '''
        ret: dict[str, any] = {}
        if (len(target) != len(node.attribute)):
            print('ERROR: diff number, node.name:',
                  node.name, ', ', end='')
            print('node.attributes:', node.attribute)
            print('target', target)
            os._exit(-1)

        for attr in node.attribute:
            attr_name = attr.name
            if (attr_name not in target):
                print('ERROR: not in target, node.name:',
                      node.name, ', ', end='')
                print(' target:', target)
                print(' node.attribute:', node.attribute)
                os._exit(-1)
            else:
                # onnx.tensor.type :
                # 1:float, 2:int, 3:string, 6:floats, 7:ints, 8:strings,
                if (attr.type == 2):   # int
                    ret[attr_name] = int(attr.i)
                elif (attr.type == 7):  # ints
                    ret[attr_name] = list(attr.ints)
                elif (attr.type == 3):  # string
                    ret[attr_name] = str(attr.s, encoding='utf-8')
                elif (attr.type == 1):  # float
                    ret[attr_name] = float(attr.f)
                else:
                    print('ERROR::Type:', attr.type, ' is not supported.')
                    os._exit(-1)
        return ret

    def __parseOnnxModel(self, onnx_model) -> None:
        '''
        解析 onnx 模型分为以下几步：
        1. 收集input, output, node, 和tensor
        2. 构建node和tensor的图关系, 即建立 node->tensor->....->tensor 的图, 使用双向链表结构记录图关系
        3. 对node进行拓扑排序, 保证 node[x] 在运行时其输入 tensor 都已经准备好
        4. 根据创建各个node对应的算子, 并初始化各算子的 attributes 和在initializer中的input
        * 实际上步骤3.和4.没有前后关系, 因为步骤4.只是新建算子与步骤3.没有联系, 只要在forward中严格根据拓扑顺序运行即可
        '''

        # 1. 收集input,output,node和tensor
        self.input = onnx_model.graph.input[0].name
        self.output = onnx_model.graph.output[0].name

        tensors: set = set()  # 所有 input 和 output 的 tensor
        initializer_tensor: dict[str, torch.Tensor] = {}
        tensor_dims: dict[str, list[int]] = {}
        tensor_pre: dict[str, set] = {}  # tensor 的前置 node
        self.tensor_next = {}  # tensor 的后置 node

        # 已经准备好的 tensor, 初始值为 initializer 中的 tensor;
        ready_tensor: set[str] = set()

        nodes: set[str] = set()   # 节点名字
        self.node_pre = {}  # node 的前置(输入) tensor
        self.node_next = {}  # node 的后置(输出) tensor

        # 假设 只有一个 input
        ready_tensor.add(self.input)
        tensor_dims[self.input] = [
            _.dim_value for _ in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        tensor_dims[self.output] = [
            _.dim_value for _ in onnx_model.graph.output[0].type.tensor_type.shape.dim]

        initializer_tensor[self.input] = torch.randn(
            tensor_dims[self.input])

        # 去除 node name 中的 '.' 因为在推理时node的名字不能包含'.'操作符号(可能是版本的原因)
        for node in onnx_model.graph.node:
            if '.' in node.name:
                node.name = node.name.replace('.', '_')

        # 初始化 tensor
        for init in onnx_model.graph.initializer:
            ready_tensor.add(init.name)
            initializer_tensor[init.name] = torch.tensor(
                onnx.numpy_helper.to_array(init))

        for node in onnx_model.graph.node:
            nodes.add(node.name)
            self.node_type[node.name] = node.op_type
            # 收集所有的 tensor
            for _ in node.input:
                tensors.add(_)
        # 获取各个 tensor 的 dims
        # (除initializer中tensor, 因为这个tensor.dim主要用于构建conv算子时的输入参数中的通道数C, 因此只需要node的实际输入/输出tensor即可)
        for node_info in onnx_model.graph.value_info:
            tensor_dims[node_info.name] = [
                _.dim_value for _ in node_info.type.tensor_type.shape.dim]

        # 2. 建立 node 和 tensor 之间的关系
        for node in onnx_model.graph.node:
            node_name = node.name
            for in_tensor in node.input:
                if (in_tensor in self.tensor_next):
                    self.tensor_next[in_tensor].add(node_name)
                else:
                    self.tensor_next[in_tensor] = {node_name}
                if (node_name in self.node_pre):
                    self.node_pre[node_name].append(in_tensor)
                else:
                    self.node_pre[node_name] = [in_tensor]

            for out_tensor in node.output:
                if (out_tensor in tensor_pre):
                    tensor_pre[out_tensor].add(node_name)
                else:
                    tensor_pre[out_tensor] = {node_name}
                if (node_name in self.node_next):
                    self.node_next[node_name].append(out_tensor)
                else:
                    self.node_next[node_name] = [out_tensor]

        # 3. 对node进行拓扑排序
        que = Queue()

        def nodeIsReady(node) -> bool:
            for pre_tensor in self.node_pre[node]:
                if (pre_tensor not in ready_tensor):
                    return False
            return True

        # 初始化 que
        for node in nodes:
            if (nodeIsReady(node)):
                que.put(node)

        while not que.empty():
            node = que.get()
            self.nodes_sorted.append(node)
            for out_tensor in self.node_next[node]:
                ready_tensor.add(out_tensor)
            for out_tensor in self.node_next[node]:
                if (out_tensor not in self.tensor_next):
                    continue
                for next_node in self.tensor_next[out_tensor]:
                    if (nodeIsReady(next_node)):
                        que.put(next_node)

        # 4. 创建各个node对应的算子
        for node in onnx_model.graph.node:
            node_name = node.name
            node_type = node.op_type
            op = nn.Module()
            if (node_type == 'Conv'):
                # 新建 op
                target = ['dilations', 'group',
                          'kernel_shape', 'pads', 'strides']
                attr: dict[str, any] = self.__getAttr(node, target)
                input_x_name = [_ for _ in self.node_pre[node_name]
                                if (not _.endswith('weight') and not _.endswith('bias'))][0]
                output_x_name = list(self.node_next[node_name])[0]
                in_ch: int = tensor_dims[input_x_name][1]
                out_ch: int = tensor_dims[output_x_name][1]

                op = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=attr['kernel_shape'][:2],
                               stride=attr['strides'], padding=attr['pads'][0:2], dilation=attr['dilations'], groups=attr['group'])

                # 初始化 op 的 weight, bias
                weight_name = [_ for _ in self.node_pre[node_name]
                               if _.endswith('weight')][0]
                bias_name = [_ for _ in self.node_pre[node_name]
                             if _.endswith('bias')][0]
                weight_tensor = initializer_tensor[weight_name]
                bias_tensor = initializer_tensor[bias_name]
                op.weight = nn.Parameter(
                    weight_tensor, requires_grad=True)
                op.bias = nn.Parameter(
                    bias_tensor, requires_grad=True)
            elif (node_type == 'MaxPool'):
                target = ['ceil_mode', 'kernel_shape',
                          'pads', 'strides']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = nn.MaxPool2d(kernel_size=attr['kernel_shape'], stride=attr['strides'],
                                  padding=attr['pads'][0:2], ceil_mode=(attr['ceil_mode'] == 1))
            elif (node_type == 'Sigmoid'):
                op = nn.Sigmoid()
            elif (node_type == 'Mul'):
                op = Mul()
                if ('Constant' in self.node_pre[node_name][1]):
                    input_factor = self.node_pre[node_name][1]
                    input_factor_tensor = initializer_tensor[input_factor]
                    op.factor = input_factor_tensor
            elif (node_type == 'Add'):
                op = Add()
                if ('Constant' in self.node_pre[node_name][1]):
                    input_factor = self.node_pre[node_name][1]
                    input_factor_tensor = initializer_tensor[input_factor]
                    op.factor = input_factor_tensor
            elif (node_type == 'Concat'):
                target = ['axis']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Concat(dim=attr['axis'])
            elif (node_type == 'Shape'):
                op = Shape()
            elif (node_type == 'Reshape'):
                input_shape = self.node_pre[node_name][1]
                input_shape_tensor = initializer_tensor[input_shape]
                op = Reshape()
                op.shape = input_shape_tensor
            elif (node_type == 'Transpose'):
                target = ['perm']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Transpose(perm=attr['perm'])
            elif (node_type == 'Split'):
                target = ['axis', 'split']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Split(axis=attr['axis'], split=attr['split'])
            elif (node_type == 'Pow'):
                op = Pow()
                input_y_name = [_ for _ in self.node_pre[node_name]
                                if 'Constant' in _][0]
                input_y_tensor = initializer_tensor[input_y_name]
                op.Y = input_y_tensor
            elif (node_type == 'Floor'):
                op = Floor()
                input_x_name = list(self.node_pre[node_name])[0]
                input_x_tensor = initializer_tensor[input_x_name]
                op.x = input_x_tensor
            elif (node_type == 'Slice'):
                op = Slice()
                input_starts_name = self.node_pre[node_name][1]
                input_starts_tensor = initializer_tensor[input_starts_name]
                input_ends_name = self.node_pre[node_name][2]
                input_ends_tensor = initializer_tensor[input_ends_name]
                input_axes_name = self.node_pre[node_name][3]

                input_axes_tensor = initializer_tensor[input_axes_name]
                op.starts = input_starts_tensor
                op.ends = input_ends_tensor
                op.axes = input_axes_tensor
            elif (node_type == 'Unsqueeze'):
                target = ['axes']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Unsqueeze(axes=attr['axes'])
            elif (node_type == 'Resize'):
                target = ['coordinate_transformation_mode',
                          'cubic_coeff_a', 'mode', 'nearest_mode']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Resize(coord_trans_mode=attr['coordinate_transformation_mode'],
                            cubic_coeff_a=attr['cubic_coeff_a'], mode=attr['mode'], nearest_mode=attr['nearest_mode'])
            elif (node_type == 'Concat'):
                target = ['axis']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Concat(dim=attr['axis'])
            elif (node_type == 'Shape'):
                op = Shape()
            elif (node_type == 'Cast'):
                target = ['to']
                attr: dict[str, any] = self.__getAttr(node, target)
                op = Cast(to=attr['to'])
            else:
                print("INFO:: not impletement: name:",
                      node_name, 'type:', node_type)
            # 把新建的算子加到图中
            self.layers_dict.add_module(node_name, op)

        return 0


def convert2Onnx(model: Model, export_onnx_path='export.onnx'):
    dynamic_input = torch.randn(32, 3, 640, 640, requires_grad=True)

    torch.onnx.export(model, dynamic_input, export_onnx_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['model_input'], output_names=['model_output'])
    print(' ')
    print('Model has been converted to ONNX')


def runOnnx(onnx_model_path='yolov3-fp32.onnx'):
    '''
    使用 onnxruntime 运行模型
    '''
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name  # 获取输入的名称
    output_name = session.get_outputs()[0].name  # 获取输出的名称
    a = 0
    input = torch.Tensor(32, 3, 640, 640).fill_(1).numpy()
    # 使用ONNXRuntime进行推断
    output = session.run([output_name], {input_name: input})
    # 处理推断结果
    print(output[0])
    pass


if __name__ == '__main__':

    model = Model('yolov3-fp32.onnx').eval()
    summary(model)

    # 使用转化的 model 运行推理(pytorch模型)
    input = torch.Tensor(32, 3, 640, 640).fill_(1)
    output = model(input)
    print(output)

    # 使用 onnxruntime 加载原始onnx模型运行推理
    # runOnnx('yolov3-fp32.onnx')

    # 导出模型
    # convert2Onnx(model, 'export_from_test.onnx')
