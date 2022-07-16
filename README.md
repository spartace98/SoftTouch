# SoftTouch: A Sensor-Placement Framework for Soft Robotic Hands
Repository containing contributed data and models described in Conference Paper Submission

## Abstract
Sensor placement for grasping tasks in conventional robotic hands has been studied, with goals including sensorizing essential contact areas or determining the effect of number of sensors on performance. However, with the new generation of dexterous soft robotic hands that deform to the shape of the object, the frameworks used to study these problems may not be sufficient.  In particular, we find that real-world experiments are essential for determining the value of different sensors and the effect of different sensor placements due to the complex interactions between the deformable robot body, sensor material properties, and sensor and task performance.  In this paper, we propose a sensor-placement framework for dexterous soft robotic hands that is easily reconfigurable to different hand designs using off-the-shelf sensors. Our three-step framework selects and evaluates candidate sensor configurations to determine the effectiveness of sensors in each configuration for estimating qualitative and quantitative manipulation metrics. We tested our framework on a soft robotic hand to select the optimum sensor placement for a given set of manipulation patterns using force and inertial sensors, which we discuss further in our submitted conference paper. 

## Folder Hierarchy
- data: Sensor readings + AprilTag readings for all the configurations
  - config1 (A1)
  - config2 (B1)
  - config3 (A2)
  - config4 (B2)
- Prediction_Models: Classifier and Regression Models to process sensor readings
  - Classifier
    - "classifier.py"
  - Regression
    - "regression.py"

## Notes
1. In the paper, we referred to configurations A1, B1, A2, and B2, which corresponds to the following subfolders in data
- config1: A1
- config2: B1
- config3: A2
- config4: B2

## Acknowledgement
- Professor Nancy: for her mentorship for this project
- Carnegie Mellon University: for providing undergraduate funding (SURF, SURG) for this project
