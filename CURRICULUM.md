# 📚 Berta Curriculum

This document outlines the complete learning architecture for Berta Chapters: a structured progression from fundamentals to advanced AI mastery.

---

## Curriculum Overview

### Mission

Democratize AI education by providing:
- **Structured learning paths** for those who want guidance
- **Self-directed learning** for those who want flexibility
- **Community-driven content** for emerging topics
- **Executable, practical knowledge** not just theory

### Who This Is For

#### 1. **The Career Changer**
*No AI background, Python novice*
- Start: Chapter 1 (Python Fundamentals)
- Path: Foundation Track (Chapters 1-5)
- Goal: Transition into AI roles

#### 2. **The Data Professional**
*Know SQL/Excel, basic Python, want to learn ML*
- Start: Chapter 1 or 6 (depending on Python comfort)
- Path: Practitioner Track (Chapters 6-15)
- Goal: Add ML expertise to existing skills

#### 3. **The Software Engineer**
*Know Python well, want to specialize in AI*
- Start: Chapter 5 or 6 (skip fundamentals)
- Path: Practitioner → Advanced
- Goal: Build AI systems professionally

#### 4. **The Researcher**
*Know ML theory, want hands-on implementation*
- Start: Chapters 9, 11, 16-18
- Path: Advanced Track (Chapters 16-25)
- Goal: Cutting-edge techniques and contributions

### Time Commitment

- **Foundation Track**: 38 hours total (~1-2 chapters per week)
- **Practitioner Track**: 88 hours total (~1-2 chapters per week)
- **Advanced Track**: 84 hours total (~1-2 chapters per week)
- **Total Curriculum**: 210+ hours to complete everything

### Prerequisites

**Absolute Minimum**:
- High school level math (algebra)
- Any programming experience (or willingness to learn)
- A laptop and internet connection

**Recommended**:
- Familiarity with Python (or willingness to learn Chapter 1)
- Comfort with college-level statistics
- Understanding of basic computer science concepts

---

## Foundation Track (Chapters 1-5)

Master the essentials. These chapters teach the core skills needed for all AI work.

### Chapter 1: Python Fundamentals for AI

**Learning Outcomes:**
- Write Python code confidently
- Understand variables, data types, and control flow
- Use functions and classes effectively
- Organize code into modules and scripts

**Topics:**
- Python syntax and best practices
- Data types: strings, numbers, lists, dictionaries, sets
- Control flow: if/elif/else, loops, comprehensions
- Functions: definition, parameters, returns, scope
- Object-oriented programming basics
- File I/O and working with external data

**Prerequisites:** None (or high school algebra)  
**Time:** 8 hours  
**Technology:** Python 3.10+, Jupyter Notebooks

---

### Chapter 2: Data Structures & Algorithms

**Learning Outcomes:**
- Choose appropriate data structures for problems
- Understand and analyze algorithm complexity
- Implement common algorithms efficiently
- Debug and optimize code

**Topics:**
- Arrays, linked lists, stacks, queues
- Trees and graphs fundamentals
- Sorting algorithms (merge sort, quicksort, etc.)
- Searching algorithms (binary search, etc.)
- Big O notation and complexity analysis
- Common algorithm patterns (recursion, dynamic programming basics)

**Prerequisites:** Chapter 1 (Python Fundamentals)  
**Time:** 6 hours  
**Technology:** Python, visualization libraries

---

### Chapter 3: Linear Algebra & Calculus for Machine Learning

**Learning Outcomes:**
- Understand vectors, matrices, and tensor operations
- Apply linear algebra to ML problems
- Understand derivatives and gradients
- Grasp the math behind neural networks

**Topics:**
- Vectors and vector operations
- Matrices and matrix operations
- Matrix decomposition (SVD, eigenvalues)
- Derivatives, partial derivatives, gradients
- Chain rule and backpropagation intuition
- Numerical optimization basics

**Prerequisites:** Chapter 1, high school algebra  
**Time:** 10 hours  
**Technology:** NumPy, visualization tools, jupyter notebooks

---

### Chapter 4: Probability & Statistics

**Learning Outcomes:**
- Reason about uncertainty using probability
- Design and interpret experiments statistically
- Understand distributions and their properties
- Apply statistical methods to data

**Topics:**
- Probability fundamentals (events, conditional probability)
- Distributions (normal, binomial, poisson, etc.)
- Bayes' theorem and Bayesian thinking
- Hypothesis testing and p-values
- Confidence intervals
- Correlation vs. causation
- A/B testing and experiment design

**Prerequisites:** Chapter 1, basic algebra  
**Time:** 8 hours  
**Technology:** Python, scipy, pandas

---

### Chapter 5: Software Design & Best Practices

**Learning Outcomes:**
- Write clean, maintainable code
- Design systems that scale
- Collaborate effectively on codebases
- Understand software engineering principles

**Topics:**
- Code organization and modularity
- Design patterns (MVC, factory, singleton, etc.)
- SOLID principles
- Testing and test-driven development
- Version control (Git) workflows
- Documentation and code comments
- Performance optimization
- Debugging techniques

**Prerequisites:** Chapter 1-2  
**Time:** 6 hours  
**Technology:** Python, Git, testing frameworks

---

## Practitioner Track (Chapters 6-15)

Apply your knowledge to real-world machine learning and AI problems.

### Chapter 6: Introduction to Machine Learning

**Learning Outcomes:**
- Understand the ML pipeline (problem → solution)
- Know the difference between supervised, unsupervised, reinforcement learning
- Build your first ML model
- Evaluate model performance correctly

**Topics:**
- ML problem framing and evaluation metrics
- Training/validation/test splits
- Supervised vs. unsupervised vs. reinforcement learning
- Bias-variance tradeoff
- Feature engineering basics
- Cross-validation
- First complete ML project

**Prerequisites:** Chapters 1-4  
**Time:** 8 hours  
**Technology:** scikit-learn, pandas, matplotlib

---

### Chapter 7: Supervised Learning - Regression & Classification

**Learning Outcomes:**
- Build regression and classification models
- Choose appropriate algorithms for problems
- Tune hyperparameters effectively
- Handle imbalanced and noisy data

**Topics:**
- Linear regression and regularization (L1, L2)
- Logistic regression
- Decision trees and ensemble methods
- Random forests and gradient boosting
- SVM and kernel methods
- Model evaluation metrics for classification and regression
- Imbalanced classification techniques

**Prerequisites:** Chapters 1-6  
**Time:** 10 hours  
**Technology:** scikit-learn, XGBoost, pandas

---

### Chapter 8: Unsupervised Learning - Clustering & Dimensionality Reduction

**Learning Outcomes:**
- Cluster data into meaningful groups
- Reduce dimensionality while preserving information
- Discover patterns in unlabeled data
- Choose and evaluate clustering solutions

**Topics:**
- K-means and hierarchical clustering
- DBSCAN and density-based methods
- Gaussian mixture models
- Principal component analysis (PCA)
- t-SNE and UMAP visualization
- Feature extraction and autoencoders basics
- Evaluation metrics for unsupervised learning

**Prerequisites:** Chapters 1-6  
**Time:** 8 hours  
**Technology:** scikit-learn, pandas, visualization libraries

---

### Chapter 9: Deep Learning Fundamentals

**Learning Outcomes:**
- Understand neural network architecture and training
- Build neural networks from scratch and using frameworks
- Apply deep learning to real problems
- Debug and optimize deep models

**Topics:**
- Artificial neurons and perceptrons
- Feedforward neural networks
- Backpropagation and gradient descent
- Activation functions and their properties
- Overfitting, regularization, and dropout
- Convolutional neural networks (CNNs)
- Introduction to PyTorch/TensorFlow
- Training loops and debugging

**Prerequisites:** Chapters 1-3, 6  
**Time:** 12 hours  
**Technology:** PyTorch or TensorFlow, NumPy

---

### Chapter 10: Natural Language Processing Basics

**Learning Outcomes:**
- Process and analyze text data
- Build NLP models for common tasks
- Understand word embeddings and representations
- Work with language datasets

**Topics:**
- Text preprocessing (tokenization, stemming, lemmatization)
- Bag of words and TF-IDF
- Word embeddings (Word2Vec, GloVe)
- Sentiment analysis
- Text classification
- Named entity recognition (NER)
- Introduction to RNNs for text
- NLP datasets and evaluation metrics

**Prerequisites:** Chapters 1-6, 9  
**Time:** 10 hours  
**Technology:** NLTK, spaCy, scikit-learn, PyTorch

---

### Chapter 11: Large Language Models & Transformers

**Learning Outcomes:**
- Understand transformer architecture deeply
- Work with modern LLMs (GPT, Claude, etc.)
- Fine-tune language models
- Build LLM-based applications

**Topics:**
- Attention mechanism and self-attention
- Transformer architecture (encoder-decoder)
- The scaling laws of language models
- Pre-training and fine-tuning paradigms
- How modern LLMs work (GPT family, etc.)
- Tokenization and vocabulary
- Generation strategies (beam search, top-k sampling)
- Using LLM APIs (OpenAI, Anthropic, open-source)

**Prerequisites:** Chapters 1, 9, 10  
**Time:** 10 hours  
**Technology:** HuggingFace Transformers, PyTorch, LLM APIs

---

### Chapter 12: Prompt Engineering & In-Context Learning

**Learning Outcomes:**
- Write effective prompts for LLMs
- Use few-shot learning effectively
- Understand LLM behavior and limitations
- Optimize LLM performance for your task

**Topics:**
- Prompt engineering principles and patterns
- Few-shot learning and in-context examples
- Chain-of-thought prompting
- System prompts and instructions
- Temperature, top-p, and generation parameters
- Avoiding hallucinations and biases
- Prompt optimization techniques
- Using LLMs as tools for various tasks

**Prerequisites:** Chapters 1, 11  
**Time:** 6 hours  
**Technology:** LLM APIs, Python

---

### Chapter 13: Retrieval-Augmented Generation (RAG)

**Learning Outcomes:**
- Build RAG systems that combine retrieval and generation
- Understand vector databases and embeddings
- Implement semantic search
- Create knowledge-grounded AI applications

**Topics:**
- Information retrieval fundamentals
- Vector embeddings and similarity search
- Vector databases (Pinecone, Weaviate, Milvus)
- RAG architecture and components
- Chunking strategies for documents
- Reranking and ranking algorithms
- Evaluation of RAG systems
- Building production RAG applications

**Prerequisites:** Chapters 1, 10, 11, 12  
**Time:** 8 hours  
**Technology:** Vector DBs, LLM APIs, embedding models

---

### Chapter 14: Fine-tuning & Adaptation Techniques

**Learning Outcomes:**
- Fine-tune models for specific domains
- Use parameter-efficient methods (LoRA, etc.)
- Adapt pre-trained models effectively
- Understand when to fine-tune vs. prompt

**Topics:**
- Full fine-tuning of language models
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Instruction tuning and RLHF overview
- Domain adaptation techniques
- Few-shot learning with fine-tuning
- Evaluation and validation strategies
- Cost-benefit analysis (fine-tune vs. prompt)
- Tools and frameworks (HuggingFace, Axolotl)

**Prerequisites:** Chapters 1, 9, 11  
**Time:** 8 hours  
**Technology:** HuggingFace, PyTorch, LoRA libraries

---

### Chapter 15: MLOps & Model Deployment

**Learning Outcomes:**
- Build reproducible ML pipelines
- Deploy models to production
- Monitor and maintain models over time
- Understand ML infrastructure

**Topics:**
- ML project structure and reproducibility
- Data pipelines and data validation
- Model versioning and experiment tracking
- CI/CD for ML (GitHub Actions, etc.)
- Containerization (Docker) for ML
- Model serving frameworks (FastAPI, TensorFlow Serving)
- Monitoring and observability
- Handling drift and retraining
- Cost optimization

**Prerequisites:** Chapters 1-6  
**Time:** 8 hours  
**Technology:** Docker, GitHub Actions, FastAPI, MLflow

---

## Advanced & Specialization Track (Chapters 16-25+)

Master complex topics and specialized domains.

### Chapter 16: Multi-Agent Systems Architecture

**Learning Outcomes:**
- Design systems with multiple AI agents
- Implement agent communication and coordination
- Build autonomous agent systems
- Understand emergent behaviors in multi-agent systems

**Topics:**
- Agent design patterns and frameworks
- Communication protocols between agents
- Coordination strategies and game theory
- Emergent behavior and swarm intelligence
- Tool use and external APIs
- Agent memory and state management
- Evaluating multi-agent systems
- Production multi-agent frameworks

**Prerequisites:** Chapters 1, 9, 11, 20  
**Time:** 10 hours  
**Technology:** LLM APIs, Agent frameworks (LangChain, AutoGPT patterns)

---

### Chapter 17: Advanced RAG & Knowledge Systems

**Learning Outcomes:**
- Build sophisticated knowledge systems
- Implement hybrid search strategies
- Design for scale and reliability
- Optimize RAG for specific domains

**Topics:**
- Hybrid search (semantic + keyword)
- Dense vs. sparse retrieval methods
- Graph-based retrieval and knowledge graphs
- Multi-hop reasoning
- Ranking and fusion strategies
- Caching and indexing optimization
- Monitoring RAG quality
- Advanced chunking and metadata strategies

**Prerequisites:** Chapters 1, 13, 17  
**Time:** 10 hours  
**Technology:** Vector DBs, Graph DBs, Neo4j

---

### Chapter 18: Reinforcement Learning Fundamentals

**Learning Outcomes:**
- Understand RL theory and algorithms
- Implement RL agents
- Apply RL to real problems
- Know the limitations and when to use RL

**Topics:**
- MDPs and Markov processes
- Temporal difference learning
- Q-learning and DQN
- Policy gradient methods (REINFORCE, A3C)
- Actor-critic methods
- RL in continuous control
- Applications and case studies
- Exploration vs. exploitation

**Prerequisites:** Chapters 1, 3, 6, 9  
**Time:** 12 hours  
**Technology:** Gym/Gymnasium, Ray RLlib, PyTorch

---

### Chapter 19: Model Optimization & Inference

**Learning Outcomes:**
- Optimize models for speed and efficiency
- Reduce model size without losing performance
- Deploy models for low-latency inference
- Understand hardware considerations

**Topics:**
- Quantization (INT8, mixed precision)
- Pruning and knowledge distillation
- Model compression techniques
- Batching and request optimization
- GPU/CPU optimization
- Inference frameworks (ONNX, TensorRT)
- Benchmarking and profiling
- Edge deployment

**Prerequisites:** Chapters 1, 9, 15  
**Time:** 8 hours  
**Technology:** PyTorch, ONNX, TensorRT

---

### Chapter 20: Building Production AI Systems

**Learning Outcomes:**
- Architect end-to-end AI systems
- Handle real-world constraints and challenges
- Design for reliability, scalability, and maintainability
- Navigate the organizational side of ML

**Topics:**
- System design principles for AI
- Handling data quality issues
- A/B testing in production
- Monitoring and debugging in production
- Cost considerations and optimization
- Documentation and knowledge sharing
- Technical debt in ML
- Cross-functional collaboration

**Prerequisites:** Chapters 1-6, 15  
**Time:** 10 hours  
**Technology:** System design patterns, tools

---

### Chapter 21: AI for Finance (Specialized)

**Learning Outcomes:**
- Apply AI techniques to financial problems
- Understand financial domain specifics
- Build trading systems and risk models
- Handle financial data correctly

**Topics:**
- Financial markets and trading concepts
- Time series forecasting for finance
- Portfolio optimization with ML
- Risk modeling and prediction
- Fraud detection
- Algorithmic trading strategies
- Backtesting and simulation
- Regulatory and ethical considerations

**Prerequisites:** Chapters 1-7, 21  
**Time:** 10 hours  
**Technology:** pandas, scikit-learn, financial APIs

---

### Chapter 22: AI Safety & Alignment

**Learning Outcomes:**
- Understand AI risks and safety concerns
- Implement safety practices
- Know alignment approaches and techniques
- Evaluate safety in your systems

**Topics:**
- AI risk taxonomy
- Alignment problem fundamentals
- Safety practices for LLMs
- Bias detection and mitigation
- Transparency and interpretability
- Red-teaming and adversarial testing
- Regulation and governance overview
- Ethical frameworks for AI

**Prerequisites:** Chapters 1, 9, 11  
**Time:** 8 hours  
**Technology:** Various tools and frameworks

---

### Chapter 23: Building Your Own AI Products

**Learning Outcomes:**
- Go from idea to AI product
- Navigate the product development process
- Understand go-to-market strategies
- Build sustainable AI businesses

**Topics:**
- AI product ideation and validation
- MVP development for AI products
- User research and feedback loops
- Monetization strategies
- Team building and hiring
- Fundraising and capital
- Customer acquisition
- Scaling and growth

**Prerequisites:** Chapters 1-6, 15, 20  
**Time:** 8 hours  
**Technology:** Product development tools

---

### Chapter 24: Research & Cutting-Edge Techniques

**Learning Outcomes:**
- Understand current research directions
- Implement recent papers
- Contribute to AI research
- Stay current with the field

**Topics:**
- Reading and implementing papers
- Current frontiers (mixture of experts, vision transformers, etc.)
- Emerging techniques and benchmarks
- Research communities and conferences
- Publishing and peer review
- Open source contributions
- Building research intuition

**Prerequisites:** Chapters 1, 9, 11, 18  
**Time:** 8 hours  
**Technology:** ArXiv, GitHub, research tools

---

### Chapter 25: AI Governance & Ethics

**Learning Outcomes:**
- Understand AI governance frameworks
- Implement ethical AI practices
- Navigate regulatory landscape
- Lead responsible AI initiatives

**Topics:**
- AI ethics frameworks
- Fairness, transparency, accountability
- Data governance and privacy
- Regulatory landscape (EU AI Act, etc.)
- Corporate governance of AI
- Stakeholder management
- Building ethical cultures
- Long-term AI strategy

**Prerequisites:** Chapters 1, 20, 22  
**Time:** 6 hours  
**Technology:** Various tools and frameworks

---

## 🎯 Learning Paths

Choose a path aligned with your goals, or create your own combination.

### Path A: "Complete AI Engineer"
Build expertise across all domains.

**Chapters**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 20

**Total Time**: ~110 hours  
**Duration**: 3-4 months at 8-10 hours/week  
**Best For**: Career changers, aspiring ML engineers, generalists  
**Outcomes**: Full-stack AI engineering, systems thinking, deployment expertise

---

### Path B: "Machine Learning Specialist"
Deep expertise in ML theory and practice.

**Chapters**: 1 → 2 → 3 → 4 → 6 → 7 → 8 → 9 → 15 → 19 → 20

**Total Time**: ~100 hours  
**Duration**: 3-4 months at 8-10 hours/week  
**Best For**: Data scientists, ML engineers, researchers  
**Outcomes**: Advanced ML techniques, optimization, production systems

---

### Path C: "LLM & NLP Expert"
Specialize in language and foundation models.

**Chapters**: 1 → 5 → 10 → 11 → 12 → 13 → 14 → 17 → 20 → 23

**Total Time**: ~90 hours  
**Duration**: 3 months at 8-10 hours/week  
**Best For**: NLP engineers, LLM application builders, prompt engineers  
**Outcomes**: LLM expertise, RAG systems, fine-tuning, production NLP

---

### Path D: "AI for Finance"
Finance-specific AI expertise.

**Chapters**: 1 → 3 → 4 → 6 → 7 → 21 → 19 → 20 → 23

**Total Time**: ~85 hours  
**Duration**: 3 months at 8-10 hours/week  
**Best For**: Finance professionals, quants, fintech engineers  
**Outcomes**: Financial ML models, trading systems, risk analysis

---

### Path E: "Quick Start: AI Fundamentals"
Fast track to functional AI knowledge.

**Chapters**: 1 → 5 → 6 → 9 → 11 → 23

**Total Time**: ~48 hours  
**Duration**: 6 weeks at 8-10 hours/week  
**Best For**: Quick learners, career explorers, busy professionals  
**Outcomes**: Core AI concepts, ability to build simple AI applications

---

### Path F: "Executive/Manager"
Understand AI at a high level for decision-making.

**Chapters**: 5 → 6 → 20 → 22 → 23 → 25

**Total Time**: ~38 hours  
**Duration**: 1-2 months at 4-5 hours/week  
**Best For**: Executives, managers, founders  
**Outcomes**: Understanding of AI capabilities, limitations, strategic thinking

---

## How to Use This Curriculum

1. **Choose your path** based on your goal and starting point
2. **Start with the first chapter** in your path
3. **Complete each chapter** by:
   - Reading the learning objectives
   - Working through the notebooks
   - Doing the exercises
   - Applying concepts to something you care about
4. **Move to the next chapter** when you feel comfortable
5. **Ask questions** via GitHub issues or by requesting custom chapters
6. **Contribute back** with improvements and new insights

---

## Requesting Custom Chapters

Can't find what you need in the planned curriculum? **Ask for it.**

[Open a Chapter Request](https://github.com/luigipascal/berta-chapters/issues/new?template=chapter-request.md)

Examples of requests we love:
- "Fine-tuning open-source models for healthcare"
- "Building chatbots for customer service"
- "Vision transformers for medical imaging"
- "Multimodal AI systems"
- "AI for climate prediction"

Berta will generate a complete chapter tailored to your need.

---

## Progress Tracking

As you complete chapters, you can:
- ⭐ Star the chapter repos
- 🔖 Bookmark your progress
- 📝 Take notes in your clones
- 🤝 Contribute back improvements

---

## Continuous Improvement

This curriculum is **living and evolving**.

- Chapters are updated based on feedback
- New chapters are added as the field advances
- Community contributions are welcome
- Issues and discussions shape the direction

---

**Created by Luigi Giacobbe | Generated by Berta AI**

*Last Updated: March 2026*
