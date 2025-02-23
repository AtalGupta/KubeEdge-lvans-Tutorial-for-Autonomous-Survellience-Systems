# Ianvs Tutorial for Edge-Based Autonomous Surveillance System with KubeEdge and Vision-Language Models

## Introduction

This tutorial demonstrates how to implement benchmarks for an **autonomous surveillance system** using Ianvs, a benchmarking tool originally designed for cloud-edge collaborative inference with LLMs, adapted here for vision-language models (VLMs). The system deploys 20 edge nodes (e.g., smart cameras) managed by KubeEdge, each processing video data locally to generate text descriptions of persons and vehicles for re-identification (ReID) and supporting natural language querying (e.g., "find the person in a red shirt"). By integrating Ianvs, we evaluate a query-routing strategy that balances lightweight edge inference with powerful cloud inference, optimizing latency, privacy, and accuracy in a distributed surveillance network.

### Why VLMs Need Cloud-Edge Collaborative Inference?

Modern VLMs, such as BLIP or LLaVA, excel at generating text from images but vary in size and capability. Small-scale VLMs (e.g., <3B parameters) can run on edge devices like smart cameras, offering low-latency and privacy-preserving inference. However, their accuracy may falter with complex images (e.g., occlusions, poor lighting), while larger VLMs (e.g., 13B+ parameters) deployed in the cloud provide superior performance at the cost of higher latency and privacy risks due to data transmission. Key challenges include:

- **Latency**: Transmitting raw images to the cloud increases Time to First Token (TTFT) for text generation, delaying ReID in time-sensitive scenarios.
- **Privacy**: Uploading video data risks exposing sensitive information, conflicting with regulations like GDPR.
- **Cost**: Cloud API calls for advanced VLMs are expensive, especially for continuous surveillance.
- **Task Variability**: Not all ReID tasks require heavy models—simple descriptions (e.g., "red car") can be handled locally.

Cloud-edge collaboration addresses these by deploying small VLMs on edge nodes for fast, private processing of easy tasks, while routing complex queries to a cloud-based large VLM, optimizing performance and resource use.

### Collaborative Inference Strategy

We adopt Ianvs’ **Query Routing** strategy, routing image queries to either an edge VLM (e.g., LLaVA-Phi) or a cloud VLM (e.g., LLaVA-13B) based on difficulty:
- Easy queries (e.g., clear images) are processed locally for speed and privacy.
- Hard queries (e.g., occluded or multi-object scenes) are sent to the cloud for accuracy.
This tutorial uses Ianvs’ `inference-then-mining` mode, where inference occurs first, followed by hard-example mining to assess difficulty and route accordingly.

## Required Resources

- **Hardware**: One machine to simulate the setup (a full 20-node deployment requires edge devices):
  - 2+ CPUs, 1 GPU (6GB+ memory), 4GB+ RAM, 10GB+ disk space.
  - For full deployment: 20 edge devices (e.g., Raspberry Pi 5 or NVIDIA Jetson).
- **Software**:
  - KubeEdge (v1.15+), Docker, Python 3.8+, PyTorch, Hugging Face Transformers.
  - Ianvs installed (see Step 1).
- **Dataset**: Combined Market-1501, VeRi-776, and PKU-ReID test set (~83,849 images, 2,334 identities) redistributed across 20 cameras (see report Task 1.2).
- **Network**: Internet for initial setup; edge nodes can operate offline post-deployment.

## Step 1. Ianvs Preparation

```bash
# Create and activate a conda environment
conda create -n ianvs-surveillance python=3.8
conda activate ianvs-surveillance

# Clone Ianvs repository
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

# Install Sedna (dependency for Ianvs)
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl

# Install example-specific dependencies
pip install -r examples/cloud-edge-collaborative-inference-for-llm/requirements.txt

# Install Ianvs core dependencies
pip install -r requirements.txt

# Install Ianvs
python setup.py install
```

## Step 2. Dataset and Model Preparation

### Dataset Configuration

1. **Prepare Dataset**:
   - Use the combined dataset from your report (Task 1.2), with images redistributed across 20 virtual cameras (e.g., ~1400 images per node for 19 nodes, ~500 for 1 node).
   - Structure it for Ianvs:
     ```
      ./dataset/surveillance-reid/
        ├── final_images/
        │   ├── c1/
        │   │   ├── v/          # images of type v
        │   │   └── p/          # images of type p
        │   ├── c2/
        │   │   ├── v/
        │   │   └── p/
        │   └── ...             # similarly for other camera nodes (up to 20)
        └── data_caption/
            ├── c1.json         # image-text pairs for camera c1
            ├── c2.json         # image-text pairs for camera c2
            └── ...             # additional caption files as needed

     ```
 

2. **Place Dataset**:
   - Move `surveillance-reid/` to `ianvs/dataset/`.
   - Update paths in `examples/surveillance-reid/testenv/testenv.yaml` if not using the default location.

### Metric Configuration

Define metrics in `examples/surveillance-reid/testenv/testenv.yaml` to evaluate performance:
- **Accuracy**: Correctness of text descriptions for ReID.
- **Edge Ratio**: Proportion of queries processed on edge.
- **TTFT**: Time to generate the first text token from an image.
- **Throughput**: Descriptions generated per second.
- **Cloud/Edge Tokens**: Number of tokens processed by each model.
