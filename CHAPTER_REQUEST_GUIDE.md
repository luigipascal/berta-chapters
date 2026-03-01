# 💬 Chapter Request Guide

Have a specific AI topic you want to learn? **Ask Berta to create a chapter for you.**

This guide explains how the chapter request system works and how to make great requests.

---

## What is a Chapter Request?

The Berta Chapters system has two streams:

### 1. **Planned Curriculum** (25+ chapters)
The structured learning path covering Python fundamentals → Advanced specializations.

### 2. **Community-Requested Chapters** (Unlimited)
Custom chapters generated on-demand based on what **you** need to learn.

This document is about Stream 2: requesting custom chapters.

---

## How It Works (In 5 Steps)

```
You Request a Topic
        ↓
    Berta Analyzes
        ↓
  Chapter Generated
        ↓
  Repository Created
        ↓
  You Learn & Contribute
```

1. **You open a GitHub issue** with your chapter request
2. **You describe** what you want to learn and why
3. **Berta AI** analyzes the request
4. **A complete chapter** is generated (notebooks, exercises, diagrams)
5. **A new repository** is created with your chapter
6. **You're notified** with a link to your chapter
7. **You learn, improve, and contribute back** ✨

---

## How to Request a Chapter

### Step 1: Go to the Issues Tab

Navigate to: [berta-chapters/issues](https://github.com/luigipascal/berta-chapters/issues)

### Step 2: Click "New Issue"

Look for the green "New Issue" button at the top right.

### Step 3: Select "Chapter Request"

You'll see a template option for "Chapter Request". Click it.

GitHub will show you a form with helpful prompts.

### Step 4: Fill Out the Form

**Required fields:**

- **Chapter Topic** (required)
  - What do you want to learn?
  - Example: "Fine-tuning LLMs for domain-specific tasks"

- **Your Experience Level** (required)
  - Select one: Beginner / Intermediate / Advanced
  - Helps Berta pitch the content at the right level

- **Learning Goal** (required)
  - Why do you need this?
  - What will you build or do with it?
  - Example: "Building a specialized chatbot for legal documents"

- **Specific Focus Areas** (optional)
  - What subtopics or angles matter most?
  - Example: "Focus on medical domain; include HIPAA considerations"

- **Preferred Tech/Tools** (optional)
  - Any specific frameworks or tools?
  - Example: "PyTorch, not TensorFlow"

- **Real-World Context** (optional)
  - Tell us about your specific use case
  - This helps Berta make the chapter more practical

### Step 5: Submit

Click "Submit new issue" and you're done!

---

## What Makes a Great Request?

### ✅ Great Requests

**Good**: "Fine-tuning LLMs for domain-specific tasks"
- ✓ Specific topic
- ✓ Clear what you want to learn

**Better**: "Fine-tuning LLMs for financial document analysis. I work in fintech and need to build a system that understands bond prospectuses. I know Python and basic ML but haven't done fine-tuning. Focus on LoRA and cost-efficiency."
- ✓ Specific topic
- ✓ Real use case
- ✓ Experience level
- ✓ Specific focus areas
- ✓ Helps Berta create relevant examples

### ❌ Vague Requests

**Avoid**: "Tell me about AI"
- ✗ Too broad
- ✗ No clear learning goal
- ✗ No context

**Avoid**: "How do I make a million dollars with AI?"
- ✗ Not a learning objective
- ✗ Not a technical topic
- ✗ Hard to create meaningful content

---

## Request Examples

Here are some real examples of great requests and how Berta would respond:

### Example 1: Vision Tasks
**Request:**
> Topic: "Building object detection systems for manufacturing defects"
> 
> Experience: Intermediate
> 
> Goal: I work in quality control for electronics manufacturing. I need to detect solder defects in circuit boards using computer vision. I know Python and scikit-learn but haven't done deep learning.
> 
> Focus: YOLO or Faster R-CNN, real-time performance, handling small objects
> 
> Tech: PyTorch preferred

**What Berta Generates:**
- Complete chapter: "Object Detection for Manufacturing Quality Control"
- Notebooks showing YOLOv8/v9 implementation
- Real circuit board image datasets (sample + links to full)
- Exercises: train detector, optimize for edge deployment
- Production script: API wrapper for manufacturing line
- Cost analysis: GPU requirements, real-time tradeoffs

---

### Example 2: Multimodal AI
**Request:**
> Topic: "Building multimodal AI systems (text + image understanding)"
> 
> Experience: Advanced
> 
> Goal: I'm building a social media content moderation system. I need to understand both the text AND images/videos in posts to flag harmful content effectively.
> 
> Focus: Vision-language models, efficiency, handling video
> 
> Tech: Open source preferred (Llama+CLIP or similar)

**What Berta Generates:**
- Chapter: "Multimodal AI Systems for Content Moderation"
- Architecture diagrams for multimodal pipelines
- Notebooks: CLIP, LLaVA, other VLMs
- Real moderation scenarios and datasets
- Exercises: build classifier, measure bias
- Production script: API for content moderation
- Performance optimization for scale

---

### Example 3: Specialized Domain
**Request:**
> Topic: "Predicting customer churn in subscription businesses"
> 
> Experience: Intermediate
> 
> Goal: I manage a SaaS product. I want to predict which customers will cancel so we can intervene.
> 
> Focus: Time series patterns, early warning signals, actionable predictions
> 
> Context: We have 2 years of customer data. Need to balance false positives (annoying customers) vs false negatives (missing churn).

**What Berta Generates:**
- Chapter: "Churn Prediction for SaaS Businesses"
- Feature engineering for subscription data
- Time series handling and window strategies
- Class imbalance techniques (important for this problem!)
- Threshold optimization (precision vs recall tradeoff)
- Production notebook: from raw data to predictions
- Business metrics: value of early intervention

---

## What Gets Generated?

Every community-requested chapter includes:

### 📄 Documentation
- README with objectives and context
- Learning outcomes matched to your goal
- Prerequisite chapters listed

### 📓 Notebooks
- **Beginner notebook**: Core concepts with simple examples
- **Intermediate notebook**: Real-world data, more complex tasks
- **Advanced notebook**: Optimization, edge cases, production considerations

### 💻 Scripts
- **main_application.py**: Production-ready implementation of what you requested
- **utilities.py**: Helper functions and reusable components
- **config.py**: Parameters and settings

### 🧪 Exercises
- 3-5 hands-on problems with increasing difficulty
- Solutions available in `solutions/` branch
- Real-world scenarios from your request

### 🎨 Diagrams
- Architecture diagrams (SVG)
- Data flow visualizations
- System design sketches
- Mermaid diagrams showing concepts

### 📊 Resources
- Links to relevant papers
- Recommended courses and tools
- Open-source repos to explore
- Dataset links

### 🗂️ Data
- Sample datasets for exercises
- Data loading scripts
- Links to full datasets if applicable

---

## Timeline Expectations

**Current Status**: All requests are processed by Berta AI.

- **Small/Focused Requests**: Generated within 1-3 days
- **Detailed/Complex Requests**: Generated within 1 week
- **Very Complex Requests**: Generated within 1-2 weeks

We prioritize requests that:
- ✅ Are detailed and specific
- ✅ Have real use cases
- ✅ Build on existing chapters
- ✅ Fill gaps in the curriculum

---

## FAQ

### Q: Can I request anything?
**A**: Almost! We focus on AI/ML topics. We love requests for:
- ✅ Specific ML techniques
- ✅ Domain applications (finance, healthcare, manufacturing, etc.)
- ✅ Building systems (chatbots, recommendation engines, etc.)
- ✅ Specialized topics (safety, ethics, optimization, etc.)

We can't fulfill:
- ❌ Non-technical topics
- ❌ Requests for non-AI content
- ❌ Homework completion
- ❌ Code-only requests without learning objectives

### Q: What if my request is rejected?
**A**: We rarely reject requests, but if we do, we'll explain why and suggest alternatives.

Common reasons:
- Too vague (fix: add more detail)
- Off-topic (we focus on AI/ML)
- Overlaps significantly with existing chapter (we'll point you there)
- Unclear learning goal (we'll ask for clarification)

### Q: Can I contribute to my requested chapter?
**A**: **Absolutely!** Once the chapter is created:
- Fork the repository
- Make improvements
- Submit a pull request
- We'll review and merge your contributions

This is how the community improves the curriculum.

### Q: Can I request updates to existing chapters?
**A**: Yes! If you:
- Found an error
- Have a suggestion for improvement
- Want additional examples
- Want a specialized variant

Open an issue on that chapter's repository.

### Q: Do I need to pay for chapters?
**A**: No. All chapters are **100% free**.
- No paywall
- No signup required
- No tracking
- Open source license

You can use them for learning, building products, teaching, and more.

### Q: Can I use these chapters commercially?
**A**: Yes, under the MIT License. You can:
- ✅ Use for commercial projects
- ✅ Modify and redistribute
- ✅ Include in products

Just include attribution to Berta AI and Luigi Giacobbe.

### Q: Can I request a chapter and not learn it?
**A**: You can request anything, but we hope you'll use the chapters you request! The system works best when:
- You request something you'll actually learn
- You complete the exercises
- You contribute improvements
- You share feedback

### Q: How do you prioritize requests?
**A**: We consider:
1. **Clarity & detail** (detailed requests first)
2. **Fill curriculum gaps** (requests for missing topics first)
3. **Community interest** (popular topics first)
4. **Complexity** (simpler first, let us build momentum)

### Q: Can multiple people request the same chapter?
**A**: Yes! If multiple people request the same topic, that signals demand and we'll prioritize it. Please search existing requests before opening a new one.

### Q: What if I have feedback on a generated chapter?
**A**: Open an issue on the chapter's repository or comment on the original request. Feedback helps Berta improve!

---

## Community Showcase

### Recently Generated Chapters

Check back here for examples of chapters the community has requested!

---

## Pro Tips for Great Requests

1. **Be Specific**: "Fine-tuning LLMs" vs "Fine-tuning LLMs with LoRA for medical domain classification"
   - Specific request = better chapter

2. **Share Context**: The more you tell us about your situation, the more practical the chapter becomes.

3. **Link to Resources**: If you have papers, datasets, or examples, share them! Helps Berta create better content.

4. **Know Your Level**: Be honest about your experience. Berta will pitch the content appropriately.

5. **Suggest Focus Areas**: What aspects matter most to you? Examples, theory, code, performance optimization?

6. **Request Related Chapters**: If your request builds on other chapters, mention them. Helps Berta connect the dots.

7. **Be Patient**: Berta works carefully to ensure quality. A week wait for excellent content beats instant mediocrity.

8. **Give Feedback**: Once your chapter is generated, use it and share feedback. This helps Berta improve for future requests.

---

## Request Template (For Reference)

```markdown
## Chapter Topic
[What do you want to learn?]

## Your Experience Level
- [ ] Beginner (New to AI/programming)
- [ ] Intermediate (Know Python & basic ML)
- [ ] Advanced (Comfortable with ML concepts)

## Learning Goal
[Why do you need this? What will you build/do with it?]

## Specific Focus Areas
[What subtopics or angles matter most?]

## Preferred Tech/Tools
[Any specific frameworks or tools?]

## Real-World Context
[Tell Berta about your specific use case]
```

---

## Questions?

- **About the request system?** Open an issue on the [main repo](https://github.com/luigipascal/berta-chapters)
- **Have feedback?** Comment on the relevant chapter's issue
- **Want to help?** Consider contributing to existing chapters or helping review new requests

---

**Created by Luigi Giacobbe | Generated by Berta AI**

*Last Updated: March 2026*

Happy requesting! 🚀
