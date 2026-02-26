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

Ray provides the following key features:

- **Unified Distributed Runtime:** Offers actor/task APIs with resource-aware
  scheduling for GPUs and CPUs, enabling elastic, fault-tolerant workloads
  on ROCm-enabled clusters.

- **AI Libraries:** Includes Ray Train for distributed training, Ray Tune
  for hyperparameter optimization, Ray RLlib for reinforcement learning,
  and Ray Serve for scalable model serving.

- **Cluster Orchestration:** Integrates with Kubernetes and on-prem schedulers
  for autoscaling, placement groups, and isolation across multi-tenant clusters.

- **Data and Streaming:** Provides Ray Data and streaming primitives for
  efficient input pipelines, batch processing, and online inference.

- **Observability and Reliability:** Built-in metrics, logging, dashboards,
  and autoscaling policies to monitor and recover long-running jobs.

Ray is commonly used in the following scenarios:

- **Distributed Training:** Scale PyTorch and other ML workloads across
  AMD Instinct GPUs with minimal code changes.

- **Hyperparameter Tuning:** Run large HPO sweeps with efficient trial
  scheduling and early stopping.

- **Reinforcement Learning:** Train RL agents at scale using RLlib with
  flexible environment integration.

- **Model Serving Pipelines:** Deploy low-latency inference services and
  batch scoring jobs with Ray Serve.

For more use cases and recommendations, see the AMD GPU tabs in the `Accelerator Support 
topic <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#accelerator-support>`__ 
of the Ray core documentation and refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/>`__, 
where you can search for Ray examples and best practices to optimize your workloads on AMD GPUs.

Why Ray?
====================================================================

Ray is well suited for end-to-end ML systems for the following reasons:

- Its **simple programming model** abstracts away distributed systems
  complexity while exposing fine-grained resource control.

- **Rich library ecosystem** accelerates common ML tasks from training
  to tuning and serving.

- **Production-grade orchestration** with autoscaling, fault tolerance,
  and observability supports enterprise deployments.

- **Seamless GPU integration** allows efficient scheduling and utilization
  of ROCm-powered AMD Instinct clusters.
