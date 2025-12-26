# GarlicBird(蒜鸟) DeepLearning Learning Note
## What is this repo?

Two old friends with backgrounds outside of AI(Physics and Chemistry, so basically science nuts) trying to learn the machine learning area step by step. We decide to reproduce the very classic papers to learn. And we find it nice to share the learning path and learning material online. We know that there is tons of learning tutorials, and our ones is nothing special to those. However, it never hurts to share(that's the spirit). 

### Our reading plan

**For the deep learning part:**

1. LeNet-5
2. AlexNet and ResNet
3. LSTM and transformer
4. GANs
5. more...(deciding after we learn and reproduce the result)

## Road maps

### Road maps of CNN



```mermaid
graph TD
    %% --- 样式定义 ---
    classDef titleClass fill:#f9f,stroke:#333,stroke-width:2px,font-size:20px,font-weight:bold;
    
    %% 早期
    classDef epoch1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    %% 爆发期
    classDef epoch2 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    %% 深层期
    classDef epoch3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    %% 变革期
    classDef epoch4 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% --- 标题 ---
    Title["CNN 进化史"]:::titleClass
    Title --- LeNet

    %% --- 结构定义 ---
    subgraph Era1 ["启蒙时代"]
        LeNet("<b>1998: LeNet-5</b><br/><i>开山鼻祖</i><br/>特点: 卷积+池化+全连接<br/>贡献: 定义了CNN的基本架构"):::epoch1
    end

    LeNet -->|"沉寂十余年后"| AlexNet

    subgraph Era2 ["深度学习爆发"]
        direction TB
        AlexNet("<b>2012: AlexNet</b><br/><i>王者归来</i><br/>特点: ReLU, Dropout, GPU<br/>贡献: ImageNet冠军，引爆DL热潮"):::epoch2
        
        AlexNet --> VGG
        AlexNet --> GoogLeNet
        
        VGG("<b>2014: VGG-16/19</b><br/><i>暴力美学</i><br/>特点: 小卷积核 3x3, 模型更深<br/>贡献: 证明了'深'的重要性"):::epoch2
        
        GoogLeNet("<b>2014: GoogLeNet</b><br/><i>拓宽视野</i><br/>特点: Inception模块, 多尺度<br/>贡献: 平衡宽度与参数量"):::epoch2
    end

    VGG --> ResNet
    GoogLeNet --> ResNet

    subgraph Era3 ["深度的极致"]
        ResNet("<b>2015: ResNet</b><br/><i>残差革命</i><br/>特点: Skip Connection 残差连接<br/>贡献: 解决梯度消失，突破百层"):::epoch3
        
        ResNet --> DenseNet
        DenseNet("<b>2017: DenseNet</b><br/><i>特征复用</i><br/>特点: 密集连接<br/>贡献: 极致的特征传递"):::epoch3
    end

    ResNet -.-> ViT

    subgraph Era4 ["从卷积到注意力"]
        ViT("<b>2020: ViT (Vision Transformer)</b><br/><i>范式转移</i><br/>特点: Patch + Self-Attention<br/>贡献: 打破CNN垄断，引入全局视野"):::epoch4
    end

    %% --- 隐藏标题连线 ---
    linkStyle 0 stroke-width:0px;
```

### Road map of RNN

```mermaid
graph TD
    %% --- 样式定义 ---
    classDef titleClass fill:#e1bee7,stroke:#4a148c,stroke-width:2px,font-size:20px,font-weight:bold;
    
    %% 史前/基础期
    classDef epoch1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    %% 黄金期 (门控机制)
    classDef epoch2 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    %% 应用爆发期 (Seq2Seq)
    classDef epoch3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    %% 终结者 (Transformer)
    classDef epoch4 fill:#ffebee,stroke:#c62828,stroke-width:2px;

    %% --- 标题 ---
    Title["RNN 进化史: 时间序列的记忆之旅"]:::titleClass
    Title --- VanillaRNN

    %% --- 结构定义 ---
    subgraph Era1 ["短期记忆时代"]
        VanillaRNN("<b>1990: Vanilla RNN (Elman)</b><br/><i>循环的起点</i><br/>特点: 简单的循环连接<br/>缺陷: 梯度消失/爆炸，<br/>只能记住极短的信息"):::epoch1
    end

    VanillaRNN -->|"苦苦支撑长序列"| LSTM
    VanillaRNN --> BiRNN

    subgraph Era2 ["门控机制的黄金时代"]
        direction TB
        LSTM("<b>1997: LSTM</b><br/><i>长短期记忆王者</i><br/>特点: 引入细胞状态(Cell State) + 3个门<br/>贡献: 彻底解决梯度消失，<br/>统治NLP领域20年"):::epoch2
        
        BiRNN("<b>1997: Bi-RNN</b><br/><i>双向视野</i><br/>特点: 正向+反向两个RNN<br/>贡献: 同时利用上下文信息"):::epoch2

        LSTM --> GRU
        LSTM --> Seq2Seq
        
        GRU("<b>2014: GRU</b><br/><i>高效的挑战者</i><br/>特点: 2个门(重置/更新)，无独立细胞状态<br/>贡献: 效果接近LSTM但参数更少，<br/>训练更快"):::epoch2
    end

    subgraph Era3 ["序列到序列的革命"]
        Seq2Seq("<b>2014: Seq2Seq (Encoder-Decoder)</b><br/><i>机器翻译架构</i><br/>特点: 变长输入到变长输出<br/>贡献: 谷歌翻译的核心，<br/>但受限于固定长度向量"):::epoch3
        
        Seq2Seq --> Attention
    end

    subgraph Era4 ["霸主的黄昏与新生"]
        Attention("<b>2015: Attention Mechanism</b><br/><i>注意力机制</i><br/>特点: 聚焦输入的特定部分<br/>贡献: 解决了Seq2Seq的长句瓶颈"):::epoch3

        Attention -.-> Transformer
        
        Transformer("<b>2017: Transformer</b><br/><i>RNN的终结者</i><br/>特点: 纯Attention，完全抛弃循环<br/>贡献: 并行计算，开启大模型时代(BERT/GPT)"):::epoch4
    end

    %% --- 隐藏标题连线 ---
    linkStyle 0 stroke-width:0px;
```

### Road maps of Transformer & Agents

```mermaid
graph TD
    %% --- 样式定义 ---
    classDef titleClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,font-size:20px,font-weight:bold;
    
    %% 模型基石期
    classDef modelBase fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    %% 生成式爆发期
    classDef modelGen fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    %% Agent 萌芽期 (理论基础)
    classDef agentTheory fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    %% Agent 爆发期 (应用落地)
    classDef agentBoom fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% --- 标题 ---
    Title["Transformer & Agent: 从理解世界到改变世界"]:::titleClass
    Title --- Transformer

    %% --- 第一阶段：基石构建 ---
    subgraph Era1 ["Transformer 纪元 (大脑的诞生)"]
        Transformer("<b>2017: Transformer</b><br/><i>万物之源</i><br/>特点: Self-Attention<br/>贡献: 抛弃RNN，奠定大模型基础"):::modelBase
        
        Transformer --> BERT
        Transformer --> GPT1
        
        BERT("<b>2018: BERT</b><br/><i>双向理解</i><br/>特点: Encoder-only, Masked LM<br/>贡献: 统治NLP理解任务"):::modelBase
        
        GPT1("<b>2018-2019: GPT-1/2</b><br/><i>生成式预训练</i><br/>特点: Decoder-only<br/>贡献: 证明了'预测下一个词'的可行性"):::modelBase
    end

    GPT1 --> GPT3

    %% --- 第二阶段：涌现与对齐 ---
    subgraph Era2 ["LLM 爆发 (智慧的涌现)"]
        GPT3("<b>2020: GPT-3</b><br/><i>大力出奇迹</i><br/>特点: 175B参数, Few-shot<br/>贡献: 展现出惊人的上下文学习能力"):::modelGen
        
        GPT3 --> Codex
        GPT3 --> InstructGPT
        
        Codex("<b>2021: Codex</b><br/><i>代码生成</i><br/>特点: 专精代码<br/>贡献: 赋予AI逻辑与编程能力，<br/>为Agent调用工具埋下伏笔"):::modelGen

        InstructGPT("<b>2022: InstructGPT/ChatGPT</b><br/><i>指令遵循</i><br/>特点: RLHF (人类反馈强化学习)<br/>贡献: 让模型听得懂人话，<br/>成为合格的'大脑'"):::modelGen
    end

    %% --- 第三阶段：思维链与工具 ---
    %% 这里是 Agent 诞生的关键转折点
    InstructGPT --> CoT
    Codex --> ToolUse

    subgraph Era3 ["Agent 的理论基石"]
        CoT("<b>2022: CoT (思维链)</b><br/><i>内在独白</i><br/>特点: Let's think step by step<br/>贡献: 解锁复杂推理，<br/>Agent规划能力的基础"):::agentTheory
        
        ToolUse("<b>2023: Toolformer</b><br/><i>工具使用</i><br/>特点: API调用<br/>贡献: AI开始学会查日历、用计算器"):::agentTheory

        ReAct("<b>2023: ReAct 范式</b><br/><i>知行合一</i><br/>特点: Reason + Act (推理+行动)<br/>贡献: Agent 的核心循环逻辑"):::agentTheory
    end
    
    CoT --> ReAct
    ToolUse --> ReAct

    %% --- 第四阶段：Agent 自主时代 ---
    ReAct --> AutoGPT

    subgraph Era4 ["Agent 寒武纪大爆发"]
        direction TB
        AutoGPT("<b>2023: AutoGPT / BabyAGI</b><br/><i>自主智能体</i><br/>特点: 循环执行任务直到目标达成<br/>贡献: 第一次让AI自己给自己下指令"):::agentBoom
        
        AutoGPT --> GenAgents
        AutoGPT --> MultiAgent

        GenAgents("<b>2023: Generative Agents</b><br/><i>斯坦福小镇</i><br/>特点: 记忆流 + 模拟社会交互<br/>贡献: 展示了Agent具备拟人化<br/>和社会化潜力"):::agentBoom

        MultiAgent("<b>2023-2024: Multi-Agent (AutoGen/MetaGPT)</b><br/><i>多智能体协作</i><br/>特点: 角色扮演 + 团队合作<br/>贡献: 解决单体能力上限，<br/>模拟软件公司流程"):::agentBoom
    end

    %% --- 隐藏连线 ---
    linkStyle 0 stroke-width:0px;
```



