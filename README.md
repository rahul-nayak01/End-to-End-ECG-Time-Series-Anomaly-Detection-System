# End-to-End-ECG-Time-Series-Anomaly-Detection-System
End-to-End ECG Time-Series Anomaly Detection System using LSTM Autoencoder

# End-to-End ECG Time-Series Anomaly Detection System

## Overview
This project implements a production-ready, end-to-end ECG time-series anomaly detection system using an unsupervised LSTM Autoencoder. The system detects abnormal heartbeats by learning normal ECG patterns and identifying deviations using reconstruction error. The project follows industry-standard ML engineering practices, covering research experimentation, modular ML pipelines, real-time inference APIs, Dockerized deployment, CI/CD automation, and AWS cloud deployment.

## Problem Statement
Electrocardiogram (ECG) signals are continuous time-series data used to monitor heart activity. Manual ECG analysis is time-consuming and does not scale for real-time monitoring systems. Additionally, labeled abnormal heartbeats are scarce, making supervised learning approaches difficult. This project aims to automatically detect anomalous ECG heartbeats using unsupervised deep learning and deploy the solution as a scalable inference service.

## Data Description
The input data is univariate ECG time-series signal data represented as continuous numerical values. The raw data is stored in CSV format, where each row corresponds to a single ECG signal value. The dataset does not contain anomaly labels, making this an unsupervised learning problem.

## Data Processing Pipeline
The ECG signal is first loaded from CSV files and converted into NumPy arrays. The signal is then standardized using statistical normalization to ensure stable model training. Since LSTM models require fixed-length sequences, the continuous ECG signal is segmented into overlapping sliding windows of fixed length, preserving temporal dependencies while enabling batch processing.

## Model Architecture
The core model is an LSTM Autoencoder consisting of an encoder LSTM that learns compressed representations of normal ECG patterns and a decoder LSTM that reconstructs the original input sequence. The model is trained to minimize reconstruction error using Mean Squared Error loss. High reconstruction error indicates anomalous behavior.

## Training Pipeline
The training pipeline loads the preprocessed ECG sequences, trains the LSTM Autoencoder on normal data, and saves the trained model as a versioned artifact. Training is fully modular and reproducible, with configuration-driven parameters for sequence length, hidden dimensions, learning rate, and epochs.

## Evaluation and Threshold Selection
After training, reconstruction errors are computed on validation data. An anomaly detection threshold is selected using a percentile-based strategy (for example, the 95th percentile of reconstruction errors). The computed threshold is stored as a separate artifact to decouple model training from inference logic.

## Anomaly Detection Logic
During inference, incoming ECG sequences are reconstructed by the trained autoencoder. The reconstruction error is calculated and compared against the stored threshold. If the error exceeds the threshold, the heartbeat sequence is classified as anomalous.

## Inference API
The trained model and threshold are exposed via a FastAPI-based inference service. The model is loaded once at application startup to ensure low-latency predictions. The API provides a health check endpoint and a prediction endpoint that accepts ECG sequences as JSON input and returns anomaly predictions with reconstruction scores.

## Dockerization
The entire application is containerized using Docker, ensuring reproducible and portable deployments. The Docker image packages the ML pipeline, inference service, and dependencies into a single deployable unit.

## CI/CD Pipeline
A GitHub Actions-based CI/CD pipeline automates the build and deployment process. On every push to the main branch, the pipeline builds a Docker image, pushes it to Amazon ECR, and deploys the latest image to an AWS EC2 instance.

## AWS Deployment
The application is deployed on AWS EC2, with Docker images stored in Amazon ECR. The deployment pipeline ensures that every code change is automatically built, versioned, and deployed without manual intervention.

## Project Structure
The repository is structured to separate research experiments from production code. Research notebooks are stored in the research directory, while the src directory contains modular data processing, feature engineering, model training, evaluation, and inference components. The app directory contains the FastAPI application, and the artifacts directory stores trained models and thresholds.

## Technologies Used
Python 3.10, PyTorch, LSTM Autoencoder, FastAPI, Docker, GitHub Actions, AWS EC2, AWS ECR, DVC

## Key Learnings
This project demonstrates end-to-end machine learning system design, unsupervised anomaly detection for time-series data, production deployment of deep learning models, containerization, CI/CD automation, and cloud-based ML system deployment.

## Future Improvements
Potential enhancements include multivariate ECG support, model monitoring and drift detection, Kubernetes-based deployment, Prometheus and Grafana monitoring, and MLflow-based experiment tracking.
---
The input data is univariate ECG time-series signal data, stored as numerical sequences, which is transformed into fixed-length sliding windows before being fed into an LSTM Autoencoder.
Rahul R Nayak  
GitHub: https://github.com/rahul-nayak01
