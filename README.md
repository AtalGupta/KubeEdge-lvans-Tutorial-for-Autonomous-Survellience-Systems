# CNCF – KubeEdge Autonomous Surveillance System Tutorial  
*Atal Gupta, 23rd February 2025*

## Overview

This tutorial shows how to set up Ianvs for evaluating a vision-language re-identification algorithm on an edge-based surveillance system. In our use case, each of the 20 edge nodes (smart cameras) locally processes images (organized into two subtypes: visual and person-specific) and generates basic text captions using a BLIP model. These image–text pairs are then used for re-identification and natural language querying. The system leverages KubeEdge to manage distributed edge devices and Ianvs to benchmark the query-routing strategy between a lightweight edge model and a more powerful cloud model.

---

## Prerequisites

- **Hardware**:  
  - A machine (laptop or VM) for simulation  
  - (Optionally) 20 edge devices (e.g., Raspberry Pi 5, NVIDIA Jetson)  
- **Software**:  
  - KubeEdge (v1.15+), Docker, Python 3.8+, PyTorch, Hugging Face Transformers  
  - Git  
- **Dataset**:  
  - Combined surveillance test set (as described in your report) with a structure similar to:  
    ```
    ./dataset/surveillance-reid/
      ├── final_images/
      │   ├── c1/
      │   │   ├── v/          # visual images
      │   │   └── p/          # person-specific images
      │   ├── c2/
      │   │   ├── v/
      │   │   └── p/
      │   └── ...             # up to 20 cameras
      └── data_caption/
          ├── c1.json         # image–text pairs for camera c1
          ├── c2.json         # image–text pairs for camera c2
          └── ...             
    ```
- **Model Consideration**:  
  - A lightweight edge VLM (e.g., BLIP-based captioner)  
  - A cloud-based VLM (for hard cases or advanced querying)

---

## Step 1. Prepare the Environment

1. **Create and Activate a Conda Environment:**

   ```bash
   conda create -n ianvs-surveillance python=3.8
   conda activate ianvs-surveillance
   ```

2. **Clone the Ianvs Repository and Install Dependencies:**

   ```bash
   git clone https://github.com/kubeedge/ianvs.git
   cd ianvs
   ```

   Install Sedna (a dependency for Ianvs):

   ```bash
   pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl
   ```

   Install example-specific dependencies:

   ```bash
   pip install -r examples/surveillance-reid/testenv/requirements.txt
   ```

   Install Ianvs core dependencies and then Ianvs itself:

   ```bash
   pip install -r requirements.txt
   python setup.py install
   ```

---

## Step 2. Dataset Preparation

1. **Structure Your Dataset:**

   Prepare the surveillance dataset as follows:

   ```
   ./dataset/surveillance-reid/
     ├── final_images/
     │   ├── c1/
     │   │   ├── v/          # visual images
     │   │   └── p/          # person-specific images
     │   ├── c2/
     │   │   ├── v/
     │   │   └── p/
     │   └── ...             # additional camera nodes (up to 20)
     └── data_caption/
         ├── c1.json         # image–text pairs for camera c1
         ├── c2.json         # image–text pairs for camera c2
         └── ...             
   ```

2. **Place the Dataset in Ianvs:**

   Move the `surveillance-reid` folder into the `ianvs/dataset/` directory. If dataset location differs, update the paths in the configuration file at:  
   `examples/surveillance-reid/testenv/testenv.yaml`.

---

## Step 3. Model and Metric Configuration

1. **Model Configuration:**

   In the configuration file (e.g., `examples/surveillance-reid/testenv/testenv.yaml` or a dedicated algorithm config), set up your models:
   
   - **EdgeModel**: This model runs on each edge device (e.g., using a BLIP-based caption generator).
   - **CloudModel**: This model is deployed in the cloud for hard cases.

   Update parameters such as model name, inference backend, temperature, and maximum tokens. For example, you may use:
   
   ```yaml
   edge_model:
     model: "BLIP-small"
     backend: "huggingface"
     temperature: 0.8
     top_p: 0.8
     max_tokens: 512

   cloud_model:
     model: "Advanced-VLM-13B"
     temperature: 0.8
     top_p: 0.8
     max_tokens: 512
   ```

2. **Metric Configuration:**

   Define the metrics in your test environment configuration to measure:
   
   - Accuracy of re-identification (text description correctness)
   - Edge Ratio (proportion of queries processed on edge)
   - Time to First Token (TTFT)
   - Throughput (descriptions generated per second)
   - Token consumption on both edge and cloud

   These metrics are calculated automatically by Ianvs based on your configuration in `testenv.yaml`.

---

## Step 4. Running the Benchmark

1. **Start the KubeEdge Cluster:**

   Ensure your KubeEdge cluster is up and running and that your edge nodes are connected. (For a local test, a single machine simulation is acceptable.)

2. **Launch the Benchmarking Job:**

   Run the following command from the root of the Ianvs repository:

   ```bash
   ianvs -f examples/surveillance-reid/testenv/testenv.yaml
   ```

   During execution, Ianvs will:
   - Load the test dataset (image and caption pairs)
   - Route each query to either the edge or cloud model based on the difficulty determined by your hard example mining module
   - Cache results to save repeated API calls during multi-round testing

3. **Monitor the Output:**

   As the job progresses, you will see log messages indicating:
   - Loading of dataset images from each camera node
   - Inference start and finish times
   - Model-specific logs (e.g., parameters used by the edge and cloud models)

---

## Step 5. Evaluating the Results

1. **Check the Output Reports:**

   After completion, Ianvs will generate a ranking file (e.g., `rank.csv` and `selected_rank.csv`) under the `ianvs/workspace` directory. These files include:
   
   - Accuracy, Edge Ratio, TTFT, Throughput, and token metrics for each configuration tested
   - Comparative results for different router strategies (e.g., EdgeOnly, CloudOnly, Query Routing)

2. **Analyze Performance:**

   Use these results to evaluate the benefits of cloud–edge collaboration. For instance:
   
   - A higher Edge Ratio indicates more queries processed locally, reducing latency and preserving privacy.
   - Compare the TTFT and throughput metrics between different models to decide if the lightweight edge model meets your application’s needs.

3. **Iterate and Optimize:**

   Based on your findings, adjust model parameters, router configurations, or dataset distribution. Rerun the benchmarking job to compare the impact of different settings.

---

This step-by-step tutorial guides you through setting up and evaluating a vision-language re-identification algorithm on a KubeEdge-managed autonomous surveillance system using Ianvs. By preparing a realistic, distributed test dataset and configuring both edge and cloud models, you can benchmark the effectiveness of a cloud–edge collaborative inference strategy for real-time surveillance applications. For further details and updates, refer to the full tutorial documentation provided in the Ianvs repository.
