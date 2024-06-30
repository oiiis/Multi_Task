# Multi-Task Learning with ResNet-18 for Optical Aerial Image Segmentation and Classification

This repository contains the implementation of a multi-task learning model using ResNet-18 as a shared encoder. The model is designed for Optical Aerial (OA) image segmentation and classification.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project implements a multi-task learning model leveraging the ResNet-18 architecture as a shared encoder. The model is capable of performing both segmentation and classification tasks on Optical Aerial images.

## Model Architecture
The model architecture includes:
- **Shared Encoder**: ResNet-18
- **Segmentation Branch**: A decoder network for image segmentation
- **Classification Branch**: Fully connected layers for image classification

The architecture ensures that the encoder learns common features useful for both tasks, leading to improved performance and efficiency.
