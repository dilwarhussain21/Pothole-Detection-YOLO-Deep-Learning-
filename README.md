# 🚧 Pothole Detection Using Deep Learning (YOLOv8)

###  Overview :
Potholes are a major threat to road safety, vehicle maintenance, and transportation efficiency. Manual inspection methods are time-consuming and prone to errors.  
This project proposes an **AI-powered Pothole Detection System** using the **YOLOv8 object detection model**, computer vision, and deep learning to automatically detect and classify potholes from road images and video feeds.

---

###  Objectives :
- Develop a robust and accurate deep learning model for **real-time pothole detection**.  
- Automate pothole identification to support **road safety**, **infrastructure maintenance**, and **driver assistance systems**.  
- Provide a lightweight, scalable, and deployable solution with a user-friendly **Flask web interface**.

---

###  Features :
-  **Real-time detection** of potholes from images and video streams.  
-  **Flask-based web application** for uploading images/videos and viewing results.  
-  **High accuracy & speed** using YOLOv8 with data augmentation.  
-  **Scalability** for large road networks and diverse environmental conditions.  
-  **Evaluation metrics**: Precision, Recall, F1-score, Accuracy.  

---

###  Tech Stack :
- **Language:** Python 3.10  
- **Frameworks & Libraries:**  
  - YOLOv8 (Ultralytics)  
  - OpenCV  
  - Flask  
  - NumPy, TensorFlow/PyTorch  
- **Tools:** Jupyter Notebook, Google Colab, PyCharm  
- **Hardware Requirements:** i5 CPU, 8GB RAM, NVIDIA GTX 1650  

---

###  Methodology :
1. **Data Collection & Preprocessing**  
   - Curated a dataset of pothole & non-pothole road images.  
   - Applied preprocessing and augmentation (rotation, brightness adjustment, flipping).  

2. **Model Training**  
   - Trained YOLOv8 on annotated pothole datasets.  
   - Fine-tuned hyperparameters for higher accuracy.  

3. **Testing & Validation**  
   - Conducted unit, integration, and system testing.  
   - Achieved robust detection under varied road and lighting conditions.  

4. **Deployment**  
   - Integrated YOLOv8 into a Flask-based web application.  
   - Supports both **image upload** and **video processing**.  

---

###  Results :
- Achieved **high detection accuracy** in real-world test scenarios.  
- Successfully processed road images and video feeds with **low latency** (~37 ms per frame).  
- GUI enables easy interaction for end-users (image/video upload → pothole detection → confidence score).  

---

###  Future Scope :
- Extend detection to **other infrastructure issues** (cracks, bumps, manholes).  
- Integrate with **IoT-enabled smart city systems** for real-time reporting.  
- Develop **autonomous pothole repair suggestions** using AI + robotics.  

---

###  Social Impact :
- Improves **road safety** by preventing accidents.  
- Reduces **vehicle damage & repair costs**.  
- Provides **data-driven insights** for efficient infrastructure planning.  
- Contributes to **smart city development**.  

---

### 📸 Screenshots
- GUI input page :
<br><br>
![gui input page](https://github.com/user-attachments/assets/6c648470-d432-4728-84a7-ccabda907713)
<br><br>

- Detection output with bounding boxes
 <br><br>
 <img width="1598" height="809" alt="screenshot_20" src="https://github.com/user-attachments/assets/4b7bbd77-ce14-44a6-814a-dc01b323da7d" />
 <br><br>
 <img width="15" height="2" alt="screenshot_24" src="https://github.com/user-attachments/assets/9af7e629-ad1d-4c10-a4cd-9b2e0f4e3055" />
<br><br>
 <img width="1597" height="813" alt="screenshot_28" src="https://github.com/user-attachments/assets/f81867f1-6c8d-404f-89ac-5de9826debd6" />
<br><br>
 <img width="16" height="2" alt="screenshot_32" src="https://github.com/user-attachments/assets/c26c4877-d309-4143-8648-efa9b7682440" />
<br><br>
 <img width="1601" height="819" alt="screenshot_33" src="https://github.com/user-attachments/assets/e8dea7e7-25ea-47fc-a211-85b3f8cdf543" />
<br><br>

- Real-time video detection
  <br><br> 
https://github.com/user-attachments/assets/a29b6b1b-3959-4fd2-8c65-e73444c81627
 <br><br> 
https://github.com/user-attachments/assets/0cac74ed-4949-4041-9ad4-d20914f4609b
 <br><br>
https://github.com/user-attachments/assets/6a42ce40-4769-4a6a-abcb-7692b6724bc8
<br><br>

---

### 👨‍💻 Authors
- **Dilwar Hussain**  
- **Prasombha Shib Sonowal**    
