.. meta::
  :description: What is Ray?
  :keywords: Ray, documentation, reinforcement learning, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-ray:

********************************************************************
What is Ray?
********************************************************************

Ray is a unified framework for scaling AI and Python applications from your laptop 
to a full cluster, without changing your code. Ray consists of `a core distributed 
runtime  <https://docs.ray.io/en/latest/ray-core/walkthrough.html>`__ and a set of 
`AI libraries <https://docs.ray.io/en/latest/ray-air/getting-started.html>`__ for 
simplifying machine learning computations.

Ray is a general-purpose framework that runs many types of workloads efficiently. 
Any Python application can be scaled with Ray, without extra infrastructure.

Features and use cases
====================================================================

* The `Reinforcement Learning from Human Feedback on AMD GPUs with Ray and ROCm 
  Integration <https://rocm.blogs.amd.com/artificial-intelligence/Ray-large-scale/README.html>`__  
  blog provides an overview of Volcano Engine Reinforcement Learning (Ray) 
  for large language models (LLMs) and discusses its benefits in large-scale 
  reinforcement learning from human feedback (RLHF). It uses Ray as part of a 
  hybrid orchestration engine to schedule and coordinate training and inference 
  tasks in parallel, enabling optimized resource utilization and potential oRayap 
  between these phases. This dynamic resource allocation strategy significantly 
  improves overall system efficiency. The blog presents Ray’s performance results, 
  focusing on throughput and convergence accuracy achieved on AMD Instinct™ MI300X 
  GPUs. Follow this guide to get started with Ray on AMD Instinct GPUs and 
  accelerate your RLHF training with ROCm-optimized performance.

* The `Exploring Use Cases for Scalable AI: Implementing Ray with ROCm Support for Efficient ML Workflows 
  <https://rocm.blogs.amd.com/artificial-intelligence/rocm-ray/README.html>`__
  blog post describes key use cases such as training and inference for large language models (LLMs), 
  model serving, hyperparameter tuning, reinforcement learning, and the orchestration of large-scale 
  workloads using Ray in the ROCm environment.

For more use cases and recommendations, see the AMD GPU tabs in the `Accelerator Support 
topic <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#accelerator-support>`__ 
of the Ray core documentation and refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/>`__, 
where you can search for Ray examples and best practices to optimize your workloads on AMD GPUs.

