# Harness Engineering CLI — Vision

## Product Vision

A model-agnostic, autonomous software-engineering harness that allows developers and researchers to give an engineering goal to an AI system and have the system plan, inspect, implement, execute, verify, evaluate, recover from failures, and deliver the result.

## Core Principle

The product must not be designed around a single LLM provider.

The LLM is a replaceable reasoning component.

The harness is the product.

## Engineering Workflow

```
USER
↓
TASK UNDERSTANDING
↓
PLANNING
↓
CONTEXT ENGINEERING
↓
AGENT ORCHESTRATION
↓
MODEL ROUTING
↓
TOOL EXECUTION
↓
OBSERVATION
↓
VERIFICATION
↓
EVALUATION
↓
REPLAN / RECOVER
↓
DELIVER
```

The goal is reliable engineering execution rather than merely generating code.

## North Star Metric

> Percentage of real engineering tasks completed successfully with minimal human intervention while respecting security, budget, and verification constraints.

## Differentiation

1. Engineering-harness architecture
2. Model/provider independence
3. Intelligent model routing
4. Automatic fallback
5. Free-model optimization
6. Local-model support
7. Context engineering
8. Specialized agents
9. Strong verification loops
10. Permission and sandboxing
11. Reproducible agent runs
12. Observability
13. Benchmarking
14. Evaluation
15. Extensibility
16. Excellent developer UX
17. Production-grade reliability
18. Research-friendly experimentation
