<div align="center">

# Embedded Vision for Real-Time 3D Printing Defect Detection

### Edge AI · Computer Vision · Embedded Deployment · Additive Manufacturing

Real-time defect detection framework for extrusion-based 3D printing using lightweight computer vision models deployed on embedded edge hardware.

Focused on:
YOLO-based defect detection, segmentation exploration, ROI optimization, model compression, quantization, multi-camera monitoring, and embedded deployment for additive manufacturing monitoring.

</div>

---

# Project Motivation

Supportless and multi-axis additive manufacturing introduces significant challenges in process monitoring and defect management.

This project focused specifically on real-time detection of:

- Stringing
- Cracking
- Warping

during active extrusion-based 3D printing.

The work was developed as part of a broader multi-axis 3D printer development process, where changing tool orientation, unsupported geometry, and complex extrusion paths increased the likelihood of dynamic print failures.

Traditional post-process inspection often results in:
- failed prints
- material waste
- increased manufacturing time
- inconsistent print quality

The primary goal of this project was to investigate whether lightweight embedded computer vision systems could perform:

# real-time in-situ defect detection directly on embedded edge hardware

while operating under strict computational and memory constraints.

The project also explored whether detected defects could be corrected automatically through adaptive printer parameter adjustment using machine learning–based feedback strategies.

Although real-time defect detection was successfully achieved, reliable automatic correction was ultimately unsuccessful due to physical and environmental inconsistencies during printing, including:
- unstable material behavior
- lighting variation
- nozzle contamination
- thermal fluctuations
- inconsistent extrusion dynamics

This repository documents both the successful deployment pipeline and the failed experimental attempts, which were equally important to the engineering and research process.

---

# System Overview

The final system integrates:

- AI camera
- Raspberry Pi edge processing
- YOLO11n lightweight detector
- ROI-based OpenCV preprocessing
- Quantized edge deployment pipeline
- Dual-camera monitoring setup
- Real-time embedded inference for additive manufacturing monitoring

The deployed pipeline performs:
1. image acquisition
2. ROI extraction
3. edge inference
4. defect localization
5. live monitoring during active printing

Target defects:
- Stringing
- Cracking
- Warping

---

# Repository Structure

```text
embedded-vision-3d-print-monitoring/
│
├── assets/
│   ├── camera_setup/
│   ├── dataset_samples/
│   └── results/
│       ├── deployment/
│       ├── segmentation/
│       └── training/
│
├── embedded_vision_system/
│   ├── deployment/
│   └── training/
│
├── experiments/
│   ├── detection_models/
│   ├── segmentation_models/
│   ├── quantization/
│   └── optimization/
│
├── model_conversion/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Dataset Engineering

A custom defect dataset was developed using:
- controlled print failures
- parameter variation
- custom defect-oriented STL models
- open-source additive manufacturing datasets
- manual annotation pipelines
- augmentation-based expansion

Dataset generation included experimentation with:
- nozzle temperature
- feed rate
- print speed
- extrusion behavior
- support structures
- geometric defect embedding

Custom STL geometries were specifically designed to induce:
- cracking
- stringing
- warping

Additional public defect datasets were integrated to improve variability and robustness.

---

# Model Exploration

Multiple lightweight architectures were explored before deployment selection.

## Detection Models

| Model | Purpose |
|---|---|
| YOLO11n | Final embedded deployment |
| YOLOv9 | Lightweight detection experiments |
| MobileNetV3 | Ultra-light edge evaluation |
| EfficientNet | Classification-oriented feature extraction |
| DETR | Transformer-based detection exploration |

## Segmentation Models

| Model | Purpose |
|---|---|
| U-Net | Pixel-level defect segmentation |
| SAM2 | Foreground isolation and segmentation experiments |

---

# Embedded Model Conversion and Optimization

A major part of the project involved converting standard deep learning models into formats compatible with constrained embedded AI hardware.

The deployment workflow included:
- model export
- optimization
- quantization
- compilation
- embedded packaging
- edge deployment validation

Multiple compression and quantization approaches were explored to reduce model size while maintaining acceptable detection accuracy.

Experiments included:
- floating-point baseline models
- mixed precision optimization
- model compression pipelines
- structured optimization techniques
- post-training quantization
- hardware-aware optimization workflows

The optimization workflow was implemented using:
- Model Compression Toolkit (MCT)
- ONNX conversion pipelines
- embedded inference compilers
- hardware-specific packaging tools

Several quantization approaches were investigated during experimentation:
- PTQ (Post-Training Quantization)
- GPTQ
- QAT (Quantization Aware Training)

Final deployment used:

# PTQ (Post-Training Quantization)

because it provided the best balance between:
- deployment simplicity
- model stability
- reduced optimization complexity
- acceptable accuracy retention
- embedded compatibility

More advanced quantization techniques such as GPTQ and QAT were explored, but introduced:
- increased optimization complexity
- longer retraining requirements
- unstable deployment behavior
- diminishing practical gains under embedded constraints

The final optimized deployment package remained approximately:

# ~6 MB

which enabled successful deployment on the embedded AI camera platform.

---

# Why YOLO11n Was Selected

Several lightweight architectures were evaluated under:
- accuracy
- latency
- model size
- embedded deployment compatibility

YOLO11n provided the best balance between:
- detection performance
- inference speed
- deployment stability

Key findings:
- larger detection models exceeded embedded deployment constraints
- lightweight classification models struggled with subtle defect differentiation
- YOLO11n maintained stable real-time inference performance while remaining deployable on embedded hardware

---

# Validation Performance

| Metric | Value |
|---|---|
| Precision | 0.785 |
| Recall | 0.870 |
| mAP@0.5 | 0.888 |
| mAP@0.5:0.95 | 0.476 |

## Per-Class Performance

| Class | Precision | Recall |
|---|---|---|
| Cracking | 0.799 | 0.967 |
| Stringing | 0.674 | 0.793 |
| Warping | 0.881 | 0.851 |

These results demonstrated reliable embedded defect detection performance under constrained hardware conditions.

---

# Embedded Deployment

The final deployment pipeline was implemented using:

- AI camera
- Raspberry Pi
- Edge inference pipeline with OpenCV preprocessing

Deployment characteristics:
- 15–20 FPS on-device inference
- 720×1080 resolution
- real-time monitoring
- ROI-based optimization
- dual-camera acquisition
- quantized embedded inference

---

# ROI-Based Optimization

A major engineering challenge involved reducing computational load on the embedded hardware.

Initial attempts using native ROI APIs produced unstable inference behavior.

To overcome this limitation, a software-based ROI preprocessing pipeline was implemented using OpenCV.

This approach:
- cropped the print region before inference
- reduced background interference
- improved detection stability
- reduced false positives
- improved real-time throughput

This became one of the most important practical engineering improvements in the project.

---

# Dual-Camera Monitoring Setup

To reduce nozzle occlusion and improve monitoring coverage, a dual-camera setup was developed using two identical embedded AI cameras positioned approximately 180° apart around the print nozzle.

Each camera used an independent custom mount and monitored the print region from opposing viewpoints.

This configuration provided:
- improved visibility around the nozzle
- reduced blind spots
- more stable defect observation
- better monitoring consistency during long-duration prints

<p align="center">
  <img src="assets/camera_setup/labeled_camera_setup.png" width="55%">
</p>

---

# Segmentation Experiments

In addition to object detection, segmentation-based pipelines were explored for:
- fine-grained defect localization
- foreground isolation
- pixel-level defect analysis

Experiments included:
- Attention U-Net
- SAM2-assisted segmentation
- combined segmentation-detection pipelines

While segmentation experiments performed well during testing, deployment on embedded hardware was ultimately infeasible because of:
- memory limitations
- quantization-related degradation
- inability to run simultaneous segmentation and detection pipelines

Despite these limitations, the experiments demonstrated the potential of segmentation-assisted monitoring for future embedded platforms.

---

# Real-Time Corrective Feedback Experiments

An experimental ML-based feedback system was explored to automatically modify printer parameters after defect detection in real time.

The objective was to create a closed-loop monitoring system capable of:
- detecting defects
- adjusting printer parameters dynamically
- reducing print failures during active fabrication

Parameters explored included:
- print speed
- extrusion behavior
- feed rate
- temperature-related adjustments

However, reliable autonomous correction proved infeasible under real-world printing conditions due to:
- unstable material behavior
- inconsistent extrusion dynamics
- thermal fluctuations
- environmental variability
- nozzle contamination
- inconsistent defect progression

Although the automatic correction pipeline was ultimately unsuccessful, the experiments provided valuable insights into the limitations of real-time adaptive control in embedded additive manufacturing systems.

The final system therefore focused on robust real-time defect detection and monitoring rather than closed-loop autonomous correction.

---

# Key Engineering Challenges

This repository intentionally documents both:

# successful deployments

and

# failed engineering attempts

because both were critical to the research process.

Major limitations encountered:
- embedded memory constraints
- segmentation deployment limitations
- quantization accuracy degradation
- unstable environmental conditions during printing
- inconsistent extrusion behavior
- thermal and lighting variability
- hardware pipeline instability during combined inference
- unreliable real-time corrective feedback behavior
- embedded deployment package restrictions

These constraints significantly influenced final system design decisions.

---

# Results

## Camera Setup

<p align="center">
  <img src="assets/camera_setup/labeled_camera_setup.png" width="55%">
</p>

---

## Edge Deployment

<p align="center">
  <img src="assets/results/deployment/final_model_running_on_camera.png" width="80%">
</p>

<p align="center">
  <img src="assets/results/deployment/results_model_run_on_system.png" width="80%">
</p>

---

## Segmentation Experiments

<p align="center">
  <img src="assets/results/segmentation/sam_on_rpi_segmented.png" width="45%">
  <img src="assets/results/segmentation/sam_on_system_result_1.png" width="45%">
</p>

---

## Training Performance

<p align="center">
  <img src="assets/results/training/yolo_training_plots.png" width="70%">
</p>

---

# Future Work

Potential future directions include:
- multi-model edge deployment
- real-time corrective feedback systems
- closed-loop print parameter optimization
- hardware acceleration for segmentation
- more capable embedded AI platforms
- temporal defect tracking
- defect severity estimation
- autonomous print correction
- multi-view defect fusion
- lightweight segmentation deployment

---

# References

1. Wickramasinghe et al. — Fused filament fabrication: State-of-the-art review  
2. An et al. — Open-source in-situ layer-wise anomaly detection  
3. Liu et al. — Closed-loop warping detection using CNNs  
4. Ronneberger et al. — U-Net: Convolutional Networks for Biomedical Image Segmentation  
5. Kirillov et al. — Segment Anything  
6. Redmon et al. — YOLO object detection architectures  
7. Howard et al. — MobileNetV3: Searching for MobileNetV3  
8. Carion et al. — End-to-End Object Detection with Transformers (DETR)