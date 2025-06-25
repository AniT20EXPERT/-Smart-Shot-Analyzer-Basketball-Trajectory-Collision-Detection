# 🏀 Smart Shot Analyzer – Basketball Trajectory & Collision Detection

A computer vision-based system that analyzes basketball shots from video footage, predicts projectile motion using polynomial regression, and detects whether the ball enters the hoop or collides with the rim.

---

## 🚀 Features

- 🎯 **Real-Time Ball Tracking** using OpenCV.
- 📈 **2nd-Degree Polynomial Regression** to predict shot trajectories.
- 🧠 **Collision Detection Logic** using $R^2$ thresholding (R² < 0.99 → rim hit).
- 🕹️ Supports **game footage** and **simulated shot videos**.
- 📊 Trajectory visualization for debugging and analysis.

---

## 🧠 Methodology

1. **Frame Extraction & Ball Detection:**  
   Detects and tracks the basketball across consecutive frames using OpenCV color filtering and contour tracking.

2. **Projectile Path Fitting:**  
   Fits a 2nd-degree polynomial curve to the tracked ball coordinates to model its parabolic motion.

3. **Collision Detection via R² Monitoring:**  
   Calculates the coefficient of determination ($R^2$) for each frame’s fitted curve.  
   - If $R^2$ drops below 0.99, it indicates a sudden path disruption → likely hoop/rim contact.

4. **Hoop Entry vs Rim Hit Classification:**  
   Analyzes the intersection of trajectory curve with hoop area. A smooth continuation is marked as a successful shot, else marked as a collision.

---

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib, SciPy  
- **Math:** Polynomial Regression, R² Error Metric  

---
