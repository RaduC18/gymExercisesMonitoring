# Gym Exercises Monitoring & Analysis System

## About the Project

This project provides a complete system for *real-time monitoring and analysis* of gym exercises and cardiovascular activity. It integrates hardware sensors, microcontroller firmware, machine-learning models, and a browser user interface.

## System Overview

To analyze motion and cardiovascular signals in real time, the project combines several core components:

* *Web Application:* A browser interface built with JavaScript and Bootstrap.
* *Firmware:* Microcontroller firmware written in Arduino C++.
* *Machine Learning:* Two independent machine learning models:
    * One for motion categorization.
    * One for PPG (photoplethysmography) signal analysis.

These parts work together to form a complete monitoring pipeline.

## Data flow through stages

1.  *ESP32 (Data Acquisition):* The microcontroller captures raw data from the sensors and transmits it via Wi-Fi.
2.  *Web Application Server:* Acts as the primary bridge, receiving the data stream from the hardware and forwarding it for analysis.
3.  *Flask Analysis Server:* A dedicated Python server that processes the incoming data using advanced algorithmic models
4.  *User (Feedback):* The analyzed results are returned to the user interface in real time.

## Training the Machine Learning Models

Two datasets were used for the machine learning components:

### Custom Motion Dataset
* *Source:* Collected from 12 individuals performing proper and improper exercise form.
* *Volume:* Around 60 labeled sequences per category, stored as CSV.
* *Design:* Designed to ensure class balance and variation across repetitions.
* *Usage:* Used to train the motion classification model.

### External PPG Dataset
* *Source:* Contains pre-labeled clinical quality heart rhythm signals.
* *Selection:* Chosen specifically to ensure high accuracy for atrial fibrillation detection.
* *Usage:* Used to train the PPG signal classification model.

## Web Interface

The web interface was created to provide a simple and user-friendly experience, especially for beginners. It supports:

* Real-time data visualization.
* Easy navigation between motion evaluation and cardiac monitoring sections.

The main page includes inspirational content and a navigation bar guiding users to the appropriate sections.
## Technical Highlights

### Embedded Programming (Arduino / ESP32)
* Real-time acquisition of motion and PPG data.
* Sensor calibration and serial/Wi-Fi communication.
* Lightweight HTTP server running on the chip for data streaming.

### Web application
* *Primary Server:* Acts as the bridge between the hardware data stream and the analysis engine.
* *Frontend:* Browser interface built with Bootstrap for responsive and user-friendly interaction.
* *Dynamic Visualization:* Real-time plotting of sensor data and exercise feedback.

### Python Analysis Backend (Flask)
* *Dedicated Processing Unit:* Flask API designed specifically to receive data from the Web App and execute complex algorithms.
* *Signal Processing:* Implementation of filtering, buffering, and Dynamic Time Warping (DTW).
* *Inference Engine:* Runs the trained ML models to classify data and return results to the web application.

### Specialized Processing Modules
* *Motion Analysis:* Combines *Dynamic Time Warping (DTW)* for sequence alignment with *Random Forest* for execution quality classification.
* *PPG (Cardio) Analysis:* Implements a hybrid *CNN (Convolutional Neural Network) + LSTM (Long Short-Term Memory)* architecture to extract spatial features and analyze temporal dependencies for Atrial Fibrillation detection.