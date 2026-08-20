from pathlib import Path

content = """# SilentSpeech AI — Project Context & Development Guide

> **Project Type:** Industry Project  
> **Purpose:** AI-powered Indian Sign Language communication system  
> **Document Role:** Single source of truth for developers, AI assistants, collaborators, and future tools working on this project.

---

# 1. Project Overview

## Project Name

**SilentSpeech AI: AI-Powered Sign Language to Speech & Text Communication System**

## Problem Statement

Millions of speech- and hearing-impaired individuals face communication barriers in educational institutions, workplaces, healthcare facilities, public services, and daily life. Communication often depends on interpreters or people who understand sign language, making interaction difficult, time-consuming, and inaccessible.

SilentSpeech AI aims to reduce this communication gap by recognizing Indian Sign Language (ISL) gestures in real time and converting them into meaningful text and speech. The system also supports speech input, converting spoken language into text to support two-way communication.

## Primary Goal

Build an end-to-end AI-powered communication platform that:

- Recognizes Indian Sign Language gestures in real time.
- Converts recognized signs into text.
- Converts generated text into speech.
- Supports speech-to-text input for reverse communication.
- Processes recognized sign tokens into meaningful sentences.
- Stores user and communication history securely.
- Provides an accessible web interface.

---

# 2. STRICT TECHNOLOGY STACK

The project must follow this technology stack unless the project team explicitly approves a change.

| Layer | Technology |
|---|---|
| Frontend | React.js |
| UI Styling | Tailwind CSS |
| Backend | Python + FastAPI |
| Database | PostgreSQL |
| Authentication | JWT |
| Computer Vision | OpenCV |
| Hand Landmark Detection | MediaPipe |
| Deep Learning | TensorFlow / Keras |
| AI Model | CNN Feature Extraction + LSTM |
| Speech-to-Text | Speech Recognition API |
| Text-to-Speech | Text-to-Speech API |
| Cloud / Deployment | Azure |
| Version Control | Git + GitHub |

## Important Rule

Do not replace the above technologies with alternatives without a documented reason and explicit approval.

Examples of changes that should NOT be made casually:

- Do not replace FastAPI with Flask or Django.
- Do not replace React with another frontend framework.
- Do not replace PostgreSQL with MongoDB.
- Do not remove MediaPipe from the hand landmark pipeline.
- Do not replace the CNN + LSTM architecture with a completely different model.
- Do not introduce a new major technology only because it is easier.

Improvements inside the approved stack are allowed.

---

# 3. SYSTEM ARCHITECTURE

## A. Sign Language Recognition Pipeline

```text
Camera
  ↓
Frame Capture
  ↓
Resize
  ↓
Color Conversion
  ↓
Preprocessing
  ↓
MediaPipe
  ↓
21 Hand Landmark Coordinates
  ↓
Convert Landmark Data into Sequence
  ↓
CNN
  ↓
Feature Extraction
  ↓
LSTM
  ↓
Sign Class + Confidence Score
  ↓
Softmax Probability
  ↓
Temporal Smoothing
  ↓
Duplicate Removal
  ↓
Confirm ISL Token
  ↓
NLP Layer
  ↓
Text Output
  ↓
Speech Output