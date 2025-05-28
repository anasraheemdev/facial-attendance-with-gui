# 🎯 Facial Attendance System with GUI

![Facial Recognition](https://www.rhombus.com/img/face-detection-recognition.png)

A modern **Facial Attendance System** built with Python and Tkinter that uses computer vision to automatically track attendance through facial recognition technology.

## ✨ Features

- 🖥️ **Intuitive GUI Interface** - User-friendly Tkinter-based graphical interface
- 👤 **Facial Recognition** - Advanced face detection and recognition capabilities
- 📊 **Attendance Tracking** - Automatic attendance logging with timestamps
- 💾 **Data Management** - Secure storage and retrieval of attendance records
- 👥 **User Management** - Easy registration and management of users
- 📈 **Real-time Processing** - Live camera feed with instant recognition
- 📋 **Export Reports** - Generate attendance reports in various formats

![Face Detection](https://espysys.com/wp-content/uploads/2024/03/Group-210.png)

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **Tkinter** - GUI framework for desktop application
- **OpenCV** - Computer vision and image processing
- **face_recognition** - Facial recognition library
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **PIL/Pillow** - Image processing

## 📁 Project Structure

```
facial-attendance-with-gui/
├── 📁 data/
│   ├── 📁 attendance/     # Attendance records
│   └── 📁 users/          # User profiles and images
├── 📄 app.py             # Main application file
├── 📄 requirements.txt   # Python dependencies
└── 📄 README.md         # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/anasraheemdev/facial-attendance-with-gui.git
   cd facial-attendance-with-gui
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
opencv-python==4.8.1.78
face-recognition==1.3.0
numpy==1.24.3
pandas==2.0.3
Pillow==10.0.0
tk==0.1.0
dlib==19.24.2
```

## 🎮 Usage

### 1. **Launch Application**
- Run `python app.py`
- The main GUI window will open

![GUI Interface](/Assets/GUI.png)

### 2. **Register New User**
- Click "Add New User" button
- Enter user details (Name, ID, etc.)
- Capture multiple face samples
- Save user profile

### 3. **Mark Attendance**
- Click "Start Attendance"
- Position face in front of camera
- System automatically recognizes and marks attendance
- View real-time attendance log

### 4. **View Reports**
- Click "View Reports" to see attendance history
- Export data to CSV or Excel format
- Filter by date range or user

![Attendance Report](https://cdn-icons-png.freepik.com/512/3135/3135823.png)

## 🔧 Configuration

### Camera Settings
```python
# In app.py, modify camera settings
CAMERA_INDEX = 0  # Change if using external camera
RECOGNITION_THRESHOLD = 0.6  # Adjust recognition sensitivity
```

### Database Settings
```python
# Configure data storage paths
ATTENDANCE_DB = "data/attendance/"
USERS_DB = "data/users/"
```

## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| Face Detection | Real-time face detection from camera | ✅ Complete |
| Face Recognition | Identify registered users | ✅ Complete |
| GUI Interface | Tkinter-based desktop app | ✅ Complete |
| Attendance Logging | Automatic attendance marking | ✅ Complete |
| User Management | Add/Edit/Delete users | ✅ Complete |
| Report Generation | Export attendance reports | ✅ Complete |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

## 📸 Screenshots

### Main Interface
![Main GUI](/Assets/main.png)

### User Registration
![User Registration](/Assets/Camera%20Face.png)

### Attendance View
![Attendance Dashboard](/Assets/attendence.png)

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Check camera permissions
   - Try different camera index (0, 1, 2...)
   - Ensure camera is not used by other applications

2. **Recognition accuracy low**
   - Ensure good lighting conditions
   - Register multiple face angles
   - Adjust recognition threshold

3. **Installation errors**
   - Update pip: `pip install --upgrade pip`
   - Install Microsoft Visual C++ Build Tools (Windows)
   - Use conda for dlib installation if pip fails

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Anas Raheem**
- GitHub: [@anasraheemdev](https://github.com/anasraheemdev)
- Email: anasraheem48@gmail.com

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- face_recognition library developers
- Tkinter for GUI framework
- All contributors and testers

## 🔮 Future Enhancements

- [ ] Web-based interface
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Multi-camera support
- [ ] Cloud storage integration
- [ ] Real-time notifications

---

**⭐ If you found this project helpful, please give it a star!**

![Star](https://cdn-icons-png.freepik.com/512/1828/1828884.png)