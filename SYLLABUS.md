# Berta Chapters - Interactive Syllabus

A visual overview of every chapter, its status, and how they connect.

---

## Curriculum Map

```mermaid
graph TD
    CH1["Ch 1: Python Fundamentals<br/>8h | Available"]
    CH2["Ch 2: Data Structures<br/>6h | Available"]
    CH3["Ch 3: Linear Algebra<br/>10h | Available"]
    CH4["Ch 4: Probability & Statistics<br/>8h | Available"]
    CH5["Ch 5: Software Design<br/>6h | Available"]

    CH6["Ch 6: Intro to ML<br/>8h | Available"]
    CH7["Ch 7: Supervised Learning<br/>10h | Available"]
    CH8["Ch 8: Unsupervised Learning<br/>8h | Available"]
    CH9["Ch 9: Deep Learning<br/>12h | Available"]
    CH10["Ch 10: NLP Basics<br/>10h | Coming Soon"]
    CH11["Ch 11: LLMs & Transformers<br/>10h | Coming Soon"]
    CH12["Ch 12: Prompt Engineering<br/>6h | Coming Soon"]
    CH13["Ch 13: RAG<br/>8h | Coming Soon"]
    CH14["Ch 14: Fine-tuning<br/>8h | Coming Soon"]
    CH15["Ch 15: MLOps<br/>8h | Coming Soon"]

    CH1 --> CH2
    CH1 --> CH3
    CH1 --> CH4
    CH2 --> CH5
    CH1 --> CH5

    CH1 --> CH6
    CH2 --> CH6
    CH3 --> CH6
    CH4 --> CH6

    CH6 --> CH7
    CH6 --> CH8
    CH3 --> CH9
    CH6 --> CH9

    CH9 --> CH10
    CH10 --> CH11
    CH11 --> CH12
    CH12 --> CH13
    CH9 --> CH14
    CH11 --> CH14
    CH6 --> CH15

    style CH1 fill:#4caf50,color:#fff
    style CH2 fill:#4caf50,color:#fff
    style CH3 fill:#4caf50,color:#fff
    style CH4 fill:#4caf50,color:#fff
    style CH5 fill:#4caf50,color:#fff
    style CH6 fill:#4caf50,color:#fff
    style CH7 fill:#4caf50,color:#fff
    style CH8 fill:#4caf50,color:#fff
    style CH9 fill:#4caf50,color:#fff
    style CH10 fill:#f3e5f5
    style CH11 fill:#f3e5f5
    style CH12 fill:#f3e5f5
    style CH13 fill:#f3e5f5
    style CH14 fill:#f3e5f5
    style CH15 fill:#f3e5f5
```

**Legend**: Green = Available | Purple = Practitioner (Coming Soon) | Chapters 1-9 fully available with SVG diagrams

---

## Chapter Status

| # | Chapter | Track | Hours | Status | Content |
|---|---------|-------|-------|--------|---------|
| 1 | [Python Fundamentals](./chapters/chapter-01-python-fundamentals/) | Foundation | 8h | Available | 3 notebooks, scripts, 6 exercises, 3 SVGs |
| 2 | [Data Structures & Algorithms](./chapters/chapter-02-data-structures/) | Foundation | 6h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 3 | [Linear Algebra & Calculus](./chapters/chapter-03-linear-algebra/) | Foundation | 10h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 4 | [Probability & Statistics](./chapters/chapter-04-probability-statistics/) | Foundation | 8h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 5 | [Software Design & Best Practices](./chapters/chapter-05-software-design/) | Foundation | 6h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 6 | [Introduction to Machine Learning](./chapters/chapter-06-intro-machine-learning/) | Practitioner | 8h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 7 | [Supervised Learning](./chapters/chapter-07-supervised-learning/) | Practitioner | 10h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 8 | [Unsupervised Learning](./chapters/chapter-08-unsupervised-learning/) | Practitioner | 8h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 9 | [Deep Learning Fundamentals](./chapters/chapter-09-deep-learning-fundamentals/) | Practitioner | 12h | Available | 3 notebooks, scripts, 5 exercises, 3 SVGs |
| 10 | Natural Language Processing | Practitioner | 10h | Planned | - |
| 11 | LLMs & Transformers | Practitioner | 10h | Planned | - |
| 12 | Prompt Engineering | Practitioner | 6h | Planned | - |
| 13 | RAG | Practitioner | 8h | Planned | - |
| 14 | Fine-tuning & Adaptation | Practitioner | 8h | Planned | - |
| 15 | MLOps & Deployment | Practitioner | 8h | Planned | - |
| 16 | Multi-Agent Systems | Advanced | 10h | Planned | - |
| 17 | Advanced RAG | Advanced | 10h | Planned | - |
| 18 | Reinforcement Learning | Advanced | 12h | Planned | - |
| 19 | Model Optimization | Advanced | 8h | Planned | - |
| 20 | Production AI Systems | Advanced | 10h | Planned | - |
| 21 | AI for Finance | Advanced | 10h | Planned | - |
| 22 | AI Safety & Alignment | Advanced | 8h | Planned | - |
| 23 | Building AI Products | Advanced | 8h | Planned | - |
| 24 | Research & Cutting-Edge | Advanced | 8h | Planned | - |
| 25 | AI Governance & Ethics | Advanced | 6h | Planned | - |

---

## What Each Chapter Includes

Every available chapter provides:

```
chapter-XX-topic/
├── README.md                     # Learning objectives and prerequisites
├── notebooks/
│   ├── 01_introduction.ipynb     # Beginner level
│   ├── 02_intermediate.ipynb     # Intermediate level
│   └── 03_advanced.ipynb         # Advanced + capstone project
├── scripts/
│   ├── main_application.py       # Production-ready code
│   └── utilities.py              # Reusable helpers
├── exercises/
│   ├── exercises.py              # Practice problems
│   └── solutions/
│       └── solutions.py          # Complete solutions
├── assets/diagrams/              # Visual references
├── datasets/                     # Sample data
└── requirements.txt              # Dependencies
```

---

## Interactive Tools

| Tool | Command | Description |
|------|---------|-------------|
| Learning Hub | `python interactive/berta.py` | Full interactive experience |
| Path Selector | `python interactive/berta.py paths` | Browse learning paths |
| Progress | `python interactive/berta.py status` | Check your progress |
| Quiz | `python interactive/berta.py quiz` | Test your knowledge |
| Chapter Info | `python interactive/berta.py chapter 1` | Get chapter details |
| New Chapter | `python templates/chapter_template.py -n 4 -t "Topic"` | Generate chapter scaffold |
| Run Tests | `python -m unittest discover -s tests` | Validate all content |

---

## Getting Started

1. **New to programming?** Start with Chapter 1
2. **Know Python?** Start with Chapter 2 or 3
3. **Not sure?** Run `python interactive/berta.py` and take the Skill Assessment

---

**Created by Luigi Pascal Rondanini | Generated by [Berta AI](https://berta.one)**

*Last Updated: March 2026*
