# InternLM2-Tutorial-Assignment-Lecture 4
# Lecture 4
# 第4课 LMDeploy 量化部署 LLM&VLM实战    
 2024.4.9 安泓郡  
 [LMDeploy](https://github.com/InternLM/LMDeploy)    
 [第4课 视频](https://www.bilibili.com/video/BV1tr421x75B/)    
 [第4课 文档](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/README.md)    
 [第4课 作业](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/homework.md)    

 ## 第4课 笔记   

### 模型部署     

- 定义：模型部署就是将训练好的深度学习模型在特定环境中运行的过程。
- 场景：
  - 服务器端：CPU部署 单GPU/TPU/NPU部署， 多卡/集卡部署
  - 移动端/边缘端：移动机器人，手机...
- 挑战：
  - 计算量巨大
  - 内存开销巨大 20B模型 40G显存
  - 访存瓶颈 大模型推理“访存密集”型任务   
  - 动态请求  请求量 时间不确定   

### 模型部署方法    

- 模型剪枝(Pruning)
- 知识蒸馏(Knowledge Distillation, KD)
- 量化(Quantization)
  
![](./LMDeploy1.png)
![](./LMDeploy2.png)
![](./LMDeploy3.png)

### **LMDeploy**   

- LMDeploy涵盖了LLM任务的全套轻量化、部署和服务解决方案；
- 核心功能：
    - 模型高效推理
    - 模型量化压缩
    - 服务化部署
- 性能提长
- 支持多模态
- LMDeploy支持多模型部署
  
![](./LMDeploy4.png)
![](./LMDeploy5.png)
![](./LMDeploy6.png)   
![](./LMDeploy7.png)
![](./LMDeploy8.png)
![](./LMDeploy9.png)

### 动手实践 - 安装、部署、量化   




    


 
  


 
 ## 第4课 作业   
 
