# Multi-Task Learning with ResNet-18 for Knee Osteoarthritis MRI Segmentation and Classification

This repository contains the implementation of a multi-task learning model using ResNet-18 as a shared encoder. The model is designed for Knee Osteoarthritis MRI segmentation and classification.

## Table of Contents1
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Description](#Description)

## Introduction
This project implements a multi-task learning model leveraging the ResNet-18 architecture as a shared encoder. The model is capable of performing both segmentation and classification tasks on Knee Osteoarthritis.

## Model Architecture
The model architecture includes:
- **Shared Encoder**: ResNet-18
- **Segmentation Branch**: A decoder network for image segmentation
- **Classification Branch**: Fully connected layers for image classification

The architecture ensures that the encoder learns common features useful for both tasks, leading to improved performance and efficiency.

## Description

This project is organized into the following structure:

- **data/**: Directory containing data files.
  - **mhd_files/**: Containing MRI files.
- **dataset.py**: Script for handling and preprocessing the dataset.
- **multi_task_model.py**: Script defining the multi-task learning model.
- **segmentation_branch.py**: Script for the segmentation branch of the model.
- **classification_branch.py**: Script for the classification branch of the model.
- **shared_encoder.py**: Script for the shared encoder used in both segmentation and classification branches.
- **train.py**: Script for training the model.



