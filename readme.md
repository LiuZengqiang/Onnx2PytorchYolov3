## onnx2pytorch for yolov3-fp.onnx
>> 一份将 yolov3-fp-32.onnx转为pytorch模型的代码
### 事先声明：本仓库不提供yolov3-fp32.onnx模型文件，因此本仓库代码只可做为将onnx模型转为pytorch模型的示范，本代码大概率无法直接应用于你的项目中！！！
## requirement
python 3.9+  
onnx  
torch  
torchinfo  
queue  
onnxruntime  

## 文件结构
``main.py``：用于加载``.onnx``模型、将模型转为pytorch模型、执行推理操作、基于转化后的torch模型导出``.onnx``模型；  
``Operator.py``：实现了.onnx模型中存在但是torch.nn中不存在的算子，例如对张量的``Pow``操作；
## 模型转化步骤：
1. 加载``.onnx``模型，收集用到的``input, output, node``和``tensor``；  
2. 构建``node``和``tensor``的图关系, 即建立``node->tensor->node->....->node->tensor``的图, 使用双向链表结构记录图关系；  
3. 对``node``进行拓扑排序, 保证推理过程中在运行``node[x]``时其输入``tensor``都已经准备好；  
4. 根据创建各个``node``对应的算子, 并初始化各算子的``attribute``和在``initializer``中的``input tensor``；  
>>注：实际上步骤3.和4.没有前后关系, 因为步骤4.只是新建算子与步骤3.没有联系, 只要在forward中严格根据拓扑顺序运行即可。