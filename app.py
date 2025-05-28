import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import cv2
import os
import datetime
import time
import numpy as np
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from PIL import Image, ImageTk

DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")
FACE_SIZE = (200, 200)  # Increase for better resolution

# directories
for directory in [DATA_DIR, USERS_DIR, ATTENDANCE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

class FaceDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        pass

class HaarCascadeDetector(FaceDetector):
    """Face detector using Haar Cascade classifier"""
    
    def __init__(self):
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.__face_cascade.empty():
            print("Error: Could not load Haar Cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is in the OpenCV data path.")
            messagebox.showerror("Error", "Could not load Haar Cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is in the OpenCV data path.")
            exit()

    def detect_faces(self, frame):
        """Detect faces in the given frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray

class FaceRecognizer(ABC):
    """Abstract base class for face recognizers"""
    
    @abstractmethod
    def compare_faces(self, face1, face2):
        """Compare two faces and return similarity score"""
        pass
    
    @abstractmethod
    def identify_user(self, face_img):
        """Identify a user from their face image"""
        pass

class AdvancedFaceRecognizer(FaceRecognizer):
    """Advanced face recognizer using multiple techniques"""
    
    def __init__(self, threshold=0.6):
        self.__threshold = threshold
        self.__sift = cv2.SIFT_create()
        
    @property
    def threshold(self):
        return self.__threshold
    
    @threshold.setter
    def threshold(self, value):
        if 0.1 <= value <= 0.9:
            self.__threshold = value
        else:
            pass

    def __extract_features(self, face_img):
        """Extract SIFT features from face image"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.equalizeHist(face_img)
        keypoints, descriptors = self.__sift.detectAndCompute(face_img, None)
        return keypoints, descriptors
    
    def __compute_lbp_histogram(self, face_img):
        """Compute Local Binary Pattern histogram"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        rows, cols = face_img.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = face_img[i, j]
                code = 0
                code |= (face_img[i-1, j-1] >= center) << 7
                code |= (face_img[i-1, j] >= center) << 6
                code |= (face_img[i-1, j+1] >= center) << 5
                code |= (face_img[i, j+1] >= center) << 4
                code |= (face_img[i+1, j+1] >= center) << 3
                code |= (face_img[i+1, j] >= center) << 2
                code |= (face_img[i+1, j-1] >= center) << 1
                code |= (face_img[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def compare_faces(self, face1, face2):
        """
        Compare two face images using multiple metrics
        Returns tuple (match_bool, similarity_score)
        """
        face1 = cv2.resize(face1, FACE_SIZE)
        face2 = cv2.resize(face2, FACE_SIZE)
        
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        lbp_hist1 = self.__compute_lbp_histogram(face1)
        lbp_hist2 = self.__compute_lbp_histogram(face2)
        lbp_score = cv2.compareHist(
            np.float32(lbp_hist1), 
            np.float32(lbp_hist2), 
            cv2.HISTCMP_CORREL
        )
        
        try:
            keypoints1, descriptors1 = self.__extract_features(face1)
            keypoints2, descriptors2 = self.__extract_features(face2)
            
            if descriptors1 is not None and descriptors2 is not None and len(keypoints1) > 10 and len(keypoints2) > 10:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                sift_score = len(good_matches) / max(len(keypoints1), len(keypoints2))
            else:
                sift_score = 0
                
        except (cv2.error, TypeError, ValueError) as e:
            sift_score = 0
            
        try:
            if face1.dtype != np.uint8:
                face1 = face1.astype(np.uint8)
            if face2.dtype != np.uint8:
                face2 = face2.astype(np.uint8)
            
            ssim_score = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)[0][0]
            ssim_score = (ssim_score + 1) / 2
        except Exception as e:
            ssim_score = 0
            
        combined_score = (
            0.35 * hist_score + 
            0.35 * lbp_score + 
            0.15 * sift_score + 
            0.15 * ssim_score
        )
        
        return combined_score >= self.__threshold, combined_score
    
    def identify_user(self, face_img):
        """
        Identify a user by comparing with registered faces
        Returns tuple (user_name, similarity_score) or (None, score)
        """
        max_similarity = -1
        identified_user = None
        
        for filename in os.listdir(USERS_DIR):
            if filename.endswith('.pkl'):
                user_path = os.path.join(USERS_DIR, filename)
                try:
                    with open(user_path, 'rb') as f:
                        user_data = pickle.load(f)
                except Exception as e:
                    continue 
                    
                username = user_data['name']
                stored_faces = user_data['faces']
                
                user_max_similarity = -1
                
                for stored_face in stored_faces:
                    if not isinstance(stored_face, np.ndarray):
                        continue
                    
                    match, similarity = self.compare_faces(face_img, stored_face)
                    user_max_similarity = max(user_max_similarity, similarity)
                
                if user_max_similarity > max_similarity:
                    max_similarity = user_max_similarity
                    identified_user = username
        
        if max_similarity >= self.__threshold:
            return identified_user, max_similarity
        else:
            return None, max_similarity

class FaceEnhancer:
    """Class for applying various face enhancement techniques"""
    def __init__(self):
        pass

    @staticmethod
    def enhance_contrast(face_img):
        """Enhance contrast using histogram equalization"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(face_img)
    
    @staticmethod
    def reduce_noise(face_img):
        """Reduce noise using Gaussian blur"""
        return cv2.GaussianBlur(face_img, (5, 5), 0) 
    
    @staticmethod
    def sharpen(face_img):
        """Sharpen image using unsharp masking"""
        blurred = cv2.GaussianBlur(face_img, (0, 0), 3) 
        return cv2.addWeighted(face_img, 1.5, blurred, -0.5, 0)
    
    @staticmethod
    def adaptive_threshold(face_img):
        """Apply adaptive thresholding"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            face_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

class AttendanceSystemGUI:
    def __init__(self, master):
        self.master = master
        master.title("Facial Attendance System")
        master.geometry("800x600")
        master.resizable(False, False)

        self.__face_detector = HaarCascadeDetector()
        self.__face_recognizer = AdvancedFaceRecognizer()
        self.__face_enhancer = FaceEnhancer()
        self.__last_marked_user = {}

        self.create_widgets()
        self.cap = None
        self.running_camera = False
        self.camera_window = None

    def create_widgets(self):
        self.title_label = tk.Label(self.master, text="Facial Attendance System", font=("Helvetica", 24, "bold"), fg="#333333")
        self.title_label.pack(pady=20)

        self.button_frame = tk.Frame(self.master, bd=2, relief="groove", padx=10, pady=10)
        self.button_frame.pack(pady=10)

        button_font = ("Helvetica", 12)
        button_width = 25
        button_height = 2

        self.mark_attendance_btn = tk.Button(self.button_frame, text="Mark Attendance", command=self.mark_attendance_gui, font=button_font, width=button_width, height=button_height, bg="#4CAF50", fg="white", activebackground="#45a049")
        self.mark_attendance_btn.grid(row=0, column=0, padx=10, pady=10)

        self.register_user_btn = tk.Button(self.button_frame, text="Register New User", command=self.register_new_user_gui, font=button_font, width=button_width, height=button_height, bg="#2196F3", fg="white", activebackground="#0b7dda")
        self.register_user_btn.grid(row=0, column=1, padx=10, pady=10)

        self.view_attendance_btn = tk.Button(self.button_frame, text="View Attendance Records", command=self.view_attendance_gui, font=button_font, width=button_width, height=button_height, bg="#FFC107", fg="black", activebackground="#ffb300")
        self.view_attendance_btn.grid(row=1, column=0, padx=10, pady=10)

        self.adjust_sensitivity_btn = tk.Button(self.button_frame, text="Adjust Recognition Sensitivity", command=self.adjust_sensitivity_gui, font=button_font, width=button_width, height=button_height, bg="#9C27B0", fg="white", activebackground="#7b1fa2")
        self.adjust_sensitivity_btn.grid(row=1, column=1, padx=10, pady=10)

        self.demonstrate_enhancement_btn = tk.Button(self.button_frame, text="Demonstrate Face Enhancement", command=self.demonstrate_enhancement_gui, font=button_font, width=button_width, height=button_height, bg="#F44336", fg="white", activebackground="#d32f2f")
        self.demonstrate_enhancement_btn.grid(row=2, column=0, padx=10, pady=10)

        self.exit_btn = tk.Button(self.button_frame, text="Exit", command=self.on_closing, font=button_font, width=button_width, height=button_height, bg="#607D8B", fg="white", activebackground="#455a64")
        self.exit_btn.grid(row=2, column=1, padx=10, pady=10)

        self.status_label = tk.Label(self.master, text="Ready", font=("Helvetica", 14), fg="blue")
        self.status_label.pack(pady=20)
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_status(self, message, color="blue"):
        # Helper function to convert RGB tuple to Tkinter hex color
        def rgb_to_hex(rgb_tuple):
            return '#%02x%02x%02x' % rgb_tuple

        # Convert the color if it's an RGB tuple
        if isinstance(color, tuple) and len(color) == 3:
            color = rgb_to_hex(color)
        
        self.status_label.config(text=message, fg=color)
        self.master.update_idletasks()

    def open_camera_window(self, title="Camera Feed", capture_mode=False, num_samples=0, enhance_capture=True):
        if self.camera_window and self.camera_window.winfo_exists():
            self.close_camera_window(self.camera_window)

        self.camera_window = tk.Toplevel(self.master)
        self.camera_window.title(title)
        self.camera_window.geometry("640x520")
        self.camera_window.resizable(False, False)
        
        self.camera_window.grid_rowconfigure(0, weight=1)
        self.camera_window.grid_columnconfigure(0, weight=1)

        self.camera_label = tk.Label(self.camera_window)
        self.camera_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        self.info_label = tk.Label(self.camera_window, text="Initializing camera...", font=("Helvetica", 12))
        self.info_label.grid(row=1, column=0, columnspan=2, pady=5)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera. Please check camera connection and permissions.")
            self.camera_window.destroy()
            return None, None
        
        self.running_camera = True
        
        capture_btn = None
        if capture_mode:
            capture_btn = tk.Button(self.camera_window, text=f"Capture Face (0/{num_samples})", 
                                    command=lambda: self.capture_face_sample(num_samples, enhance_capture, self.camera_window), 
                                    font=("Helvetica", 12), bg="#4CAF50", fg="white", width=20)
            capture_btn.grid(row=2, column=0, padx=10, pady=5)
            self.capture_btn_ref = capture_btn

        quit_btn = tk.Button(self.camera_window, text="Quit Camera", 
                             command=lambda: self.close_camera_window(self.camera_window), 
                             font=("Helvetica", 12), bg="#F44336", fg="white", width=20)
        quit_btn.grid(row=2, column=1, padx=10, pady=5)
        
        self.camera_window.protocol("WM_DELETE_WINDOW", lambda: self.close_camera_window(self.camera_window))

        return self.camera_window, capture_btn

    def close_camera_window(self, camera_window):
        self.running_camera = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if camera_window and camera_window.winfo_exists():
            camera_window.destroy()
        self.camera_window = None

    def update_frame(self, camera_window, capture_mode, num_samples, enhance_capture, capture_btn_ref, current_samples_list=None, recognition_buffer=None):
        if not self.running_camera or (camera_window and not camera_window.winfo_exists()):
            return

        ret, frame = self.cap.read()
        if not ret:
            self.info_label.config(text="Failed to grab frame. Reconnecting...", fg="red")
            camera_window.after(1000, lambda: self.update_frame(camera_window, capture_mode, num_samples, enhance_capture, capture_btn_ref, current_samples_list, recognition_buffer))
            return
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        faces, gray = self.__face_detector.detect_faces(frame)

        # Helper function to convert RGB tuple to Tkinter hex color
        def rgb_to_hex(rgb_tuple):
            return '#%02x%02x%02x' % rgb_tuple

        if capture_mode:
            if capture_btn_ref:
                capture_btn_ref.config(text=f"Capture Face ({len(current_samples_list)}/{num_samples})")
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.info_label.config(text=f"Face Detected! Samples: {len(current_samples_list)}/{num_samples}. Click 'Capture Face'.", fg=rgb_to_hex((0, 255, 0))) # Green
            else:
                self.info_label.config(text=f"No face detected. Please align your face. Samples: {len(current_samples_list)}/{num_samples}.", fg=rgb_to_hex((0, 0, 255))) # Red
        else:
            display_text = "Scanning for face..."
            text_color_rgb = (0, 0, 255) # Default Red for unknown/scanning
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                
                enhanced_face = self.__face_enhancer.enhance_contrast(face_img)
                enhanced_face = self.__face_enhancer.reduce_noise(enhanced_face)
                
                user, similarity = self.__face_recognizer.identify_user(enhanced_face)
                
                recognition_buffer.append((user, similarity))
                if len(recognition_buffer) > 15: 
                    recognition_buffer.pop(0)
                
                stable_user = None
                if len(recognition_buffer) >= 2: 
                    valid_recognitions = [r[0] for r in recognition_buffer if r[0] is not None]
                    
                    if valid_recognitions:
                        user_counts = Counter(valid_recognitions)
                        most_common_user, count = user_counts.most_common(1)[0]
                        
                        if count >= int(len(recognition_buffer) * 0.7): 
                            stable_user = most_common_user

                if stable_user:
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    attendance_file = os.path.join(ATTENDANCE_DIR, f"{date_str}.txt")
                    
                    user_already_marked = False
                    if os.path.exists(attendance_file):
                        with open(attendance_file, "r") as f:
                            next(f, None) 
                            for line in f:
                                if line.startswith(stable_user + ","):
                                    user_already_marked = True
                                    break
                    
                    if not user_already_marked:
                        if not os.path.exists(attendance_file):
                            with open(attendance_file, "w") as f:
                                f.write("Name,Time,Confidence\n")
                        
                        with open(attendance_file, "a") as f:
                            f.write(f"{stable_user},{time_str},{similarity:.2f}\n")
                        
                        display_text = f"Attendance Marked: {stable_user}!"
                        text_color_rgb = (0, 0, 255) # Blue for marked (OpenCV uses BGR, Tkinter uses RGB)
                        self.update_status(f"Attendance marked for {stable_user}!", "green") # Tkinter named color
                        self.close_camera_window(camera_window)
                        return
                    else:
                        display_text = f"{stable_user} already marked attendance today."
                        text_color_rgb = (0, 255, 255) # Yellow for already marked
                        self.update_status(f"{stable_user} already marked attendance today.", "orange") # Tkinter named color
                        
                elif user:
                    display_text = f"Recognizing: {user} ({similarity:.2f})"
                    text_color_rgb = (0, 255, 0) # Green for recognized but not stable
                else:
                    display_text = "Unknown Face"
                    text_color_rgb = (0, 0, 255) # Red for unknown
                    
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), text_color_rgb, 2) # OpenCV expects BGR
                cv2.putText(display_frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_rgb, 2)
            else:
                display_text = "No face detected"
                text_color_rgb = (0, 0, 255) # Red for no face

            # Apply Tkinter hex color for info_label
            self.info_label.config(text=f"{display_text}\nThreshold: {self.__face_recognizer.threshold:.2f}", fg=rgb_to_hex(text_color_rgb))


        img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.config(image=imgtk)

        camera_window.after(10, lambda: self.update_frame(camera_window, capture_mode, num_samples, enhance_capture, capture_btn_ref, current_samples_list, recognition_buffer))


    def capture_face_sample(self, num_samples, enhance_capture, camera_window):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera is not active.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame.")
            return
        
        frame = cv2.flip(frame, 1)
        faces, gray = self.__face_detector.detect_faces(frame)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, FACE_SIZE)
            
            if enhance_capture:
                face_img = self.__face_enhancer.enhance_contrast(face_img)
                face_img = self.__face_enhancer.reduce_noise(face_img) 

            self.current_registration_samples.append(face_img)
            self.update_status(f"Sample {len(self.current_registration_samples)}/{num_samples} captured.", "green")

            if self.capture_btn_ref:
                self.capture_btn_ref.config(text=f"Capture Face ({len(self.current_registration_samples)}/{num_samples})")

            if len(self.current_registration_samples) >= num_samples:
                self.close_camera_window(camera_window)
                messagebox.showinfo("Registration Complete", f"Captured {num_samples} face samples. Click OK to save.")
                self.save_registered_user_data()
        else:
            messagebox.showwarning("No Face", "No face detected in the frame. Please align your face.")

    def save_registered_user_data(self):
        name = self.current_registration_name
        filename = name.lower().replace(" ", "_")
        user_file = os.path.join(USERS_DIR, f"{filename}.pkl")
        
        user_data = {
            'name': name,
            'faces': []
        }

        if os.path.exists(user_file):
            try:
                with open(user_file, 'rb') as f:
                    user_data = pickle.load(f)
            except Exception as e:
                messagebox.showwarning("Load Error", f"Error loading existing user data: {e}. Starting with new data.")
        
        if self.current_registration_samples:
            user_data['faces'].extend(self.current_registration_samples)
            
            try:
                with open(user_file, 'wb') as f:
                    pickle.dump(user_data, f)
                self.update_status(f"User '{name}' registered successfully with {len(self.current_registration_samples)} new face samples!", "green")
                messagebox.showinfo("Success", f"User '{name}' registered successfully with {len(self.current_registration_samples)} new face samples!\nTotal samples: {len(user_data['faces'])}")
            except Exception as e:
                self.update_status(f"Error saving user data for {name}: {e}", "red")
                messagebox.showerror("Save Error", f"Error saving user data for {name}: {e}")
        else:
            self.update_status("Registration failed. No faces were captured.", "red")
            messagebox.showwarning("Failed", "Registration failed. No faces were captured.")
        
        self.current_registration_samples = []
        self.current_registration_name = ""


    def mark_attendance_gui(self):
        self.update_status("Starting attendance marking...", "blue")
        camera_window, _ = self.open_camera_window("Mark Attendance")
        if camera_window:
            recognition_buffer = []
            self.master.after(10, lambda: self.update_frame(camera_window, False, 0, False, None, None, recognition_buffer))

    def register_new_user_gui(self):
        name = simpledialog.askstring("Register New User", "Enter the new user's name:")
        if name:
            name = name.strip()
            if not name:
                messagebox.showwarning("Input Error", "Name cannot be empty. Registration cancelled.")
                return

            filename = name.lower().replace(" ", "_")
            user_file = os.path.join(USERS_DIR, f"{filename}.pkl")

            if os.path.exists(user_file):
                if not messagebox.askyesno("User Exists", f"User '{name}' already exists. Do you want to add more face samples to their profile?"):
                    self.update_status("Registration cancelled by user.", "orange")
                    return
            
            self.current_registration_name = name
            self.current_registration_samples = []
            
            num_samples = 5
            camera_window, capture_btn = self.open_camera_window("Register New User: Capture Faces", capture_mode=True, num_samples=num_samples, enhance_capture=True)
            if camera_window:
                self.master.after(10, lambda: self.update_frame(camera_window, True, num_samples, True, capture_btn, self.current_registration_samples))

    def view_attendance_gui(self):
        self.update_status("Viewing attendance records...", "blue")
        attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.txt')]
        
        if not attendance_files:
            messagebox.showinfo("No Records", "No attendance records found.")
            self.update_status("Ready", "blue")
            return
        
        attendance_files.sort(reverse=True)

        attendance_viewer = tk.Toplevel(self.master)
        attendance_viewer.title("Attendance Records")
        attendance_viewer.geometry("600x500")

        tk.Label(attendance_viewer, text="Select Date or 'All' to View Records:", font=("Helvetica", 14, "bold")).pack(pady=10)

        date_options = ["All Records"] + [f.split('.')[0] for f in attendance_files]
        self.selected_date = tk.StringVar(attendance_viewer)
        self.selected_date.set(date_options[0])
        
        date_menu = tk.OptionMenu(attendance_viewer, self.selected_date, *date_options)
        date_menu.config(font=("Helvetica", 12))
        date_menu.pack(pady=5)

        self.attendance_text_area = scrolledtext.ScrolledText(attendance_viewer, wrap=tk.WORD, font=("Consolas", 10))
        self.attendance_text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.attendance_text_area.config(state=tk.DISABLED)

        display_btn = tk.Button(attendance_viewer, text="Display Records", command=self._display_selected_attendance, font=("Helvetica", 12), bg="#2196F3", fg="white")
        display_btn.pack(pady=10)
        
        self._display_selected_attendance()

    def _display_selected_attendance(self):
        self.attendance_text_area.config(state=tk.NORMAL)
        self.attendance_text_area.delete(1.0, tk.END)

        choice = self.selected_date.get()
        records_to_display = []
        
        all_attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.txt')]
        all_attendance_files.sort(reverse=True)

        files_to_process = []
        if choice == 'All Records':
            files_to_process = all_attendance_files
        else:
            for filename in all_attendance_files:
                if filename.startswith(choice + "."):
                    files_to_process = [filename]
                    break

        if not files_to_process:
            self.attendance_text_area.insert(tk.END, "No attendance records found for the selected date.")
            self.attendance_text_area.config(state=tk.DISABLED)
            return

        for filename in files_to_process:
            date_from_filename = filename.split('.')[0]
            file_path = os.path.join(ATTENDANCE_DIR, filename)
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    if lines and lines[0].strip().lower() == "name,time,confidence":
                        data_lines = lines[1:]
                    else:
                        data_lines = lines

                    if not data_lines:
                        continue

                    for line in data_lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            time_recorded = parts[1].strip()
                            confidence = f"Confidence: {float(parts[2]):.2f}" if len(parts) > 2 else ""
                            records_to_display.append(f"{name} - {date_from_filename} - {time_recorded} {confidence}")
            except Exception as e:
                self.attendance_text_area.insert(tk.END, f"Error reading {filename}: {e}\n")

        if records_to_display:
            records_to_display.sort(key=lambda x: (x.split(' - ')[1], x.split(' - ')[2]))
            self.attendance_text_area.insert(tk.END, "--- Attendance Records ---\n")
            for record in records_to_display:
                self.attendance_text_area.insert(tk.END, record + "\n")
            self.attendance_text_area.insert(tk.END, "--------------------------")
        else:
            self.attendance_text_area.insert(tk.END, "No attendance records found for the selected criteria.")
        
        self.attendance_text_area.config(state=tk.DISABLED)

    def adjust_sensitivity_gui(self):
        current_threshold = self.__face_recognizer.threshold
        new_threshold_str = simpledialog.askstring("Adjust Sensitivity", 
                                                f"Current recognition threshold: {current_threshold:.2f}\n"
                                                "Enter new threshold (0.1-0.9):\n"
                                                "(Lower = more lenient, Higher = more strict)")
        if new_threshold_str:
            try:
                new_threshold = float(new_threshold_str)
                if 0.1 <= new_threshold <= 0.9:
                    self.__face_recognizer.threshold = new_threshold
                    self.update_status(f"Recognition threshold updated to {self.__face_recognizer.threshold:.2f}", "green")
                    messagebox.showinfo("Success", f"Recognition threshold updated to {self.__face_recognizer.threshold:.2f}")
                else:
                    self.update_status("Invalid value. Threshold must be between 0.1 and 0.9.", "red")
                    messagebox.showwarning("Invalid Input", "Invalid value. Threshold must be between 0.1 and 0.9.")
            except ValueError:
                self.update_status("Invalid input. Please enter a numerical value.", "red")
                messagebox.showerror("Invalid Input", "Invalid input. Please enter a numerical value.")
        else:
            self.update_status("Threshold adjustment cancelled.", "orange")

    def demonstrate_enhancement_gui(self):
        self.update_status("Capturing face for enhancement demo...", "blue")
        
        self.toggle_buttons(False)

        camera_window = tk.Toplevel(self.master)
        camera_window.title("Face Enhancement Demo - Capture")
        camera_window.geometry("640x520")
        camera_window.resizable(False, False)
        
        camera_window.grid_rowconfigure(0, weight=1)
        camera_window.grid_columnconfigure(0, weight=1)

        cam_label = tk.Label(camera_window)
        cam_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        info_label = tk.Label(camera_window, text="Looking for a face to capture for demo...", font=("Helvetica", 12))
        info_label.grid(row=1, column=0, columnspan=2, pady=5)

        captured_face_for_demo = []
        cap_demo = None # Initialize cap_demo here

        def update_demo_frame():
            nonlocal cap_demo # Declare cap_demo as nonlocal
            
            if not camera_window.winfo_exists():
                if cap_demo and cap_demo.isOpened(): # Check if cap_demo is initialized and open
                    cap_demo.release()
                self.toggle_buttons(True)
                self.update_status("Face enhancement demo cancelled.", "orange")
                return

            if not cap_demo: # If cap_demo is not yet initialized (first run)
                cap_demo = cv2.VideoCapture(0)
                if not cap_demo.isOpened():
                    info_label.config(text="Camera Error! Retrying...", fg="red")
                    camera_window.after(1000, update_demo_frame)
                    return # Exit this frame and retry
            
            if not cap_demo.isOpened(): # If camera was open but now lost
                info_label.config(text="Camera Error! Reconnecting...", fg="red")
                camera_window.after(1000, update_demo_frame)
                return

            ret, frame = cap_demo.read()
            if not ret:
                info_label.config(text="Failed to grab frame. Retrying...", fg="red")
                camera_window.after(1000, update_demo_frame)
                return

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            faces, gray = self.__face_detector.detect_faces(frame)

            # Convert OpenCV BGR colors to Tkinter hex colors
            def rgb_to_hex(rgb_tuple):
                return '#%02x%02x%02x' % rgb_tuple

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # OpenCV BGR
                cv2.putText(display_frame, "Face Found! Capturing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # OpenCV BGR
                
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                captured_face_for_demo.append(face_img)

                cap_demo.release()
                camera_window.destroy()
                self._show_enhancement_results(captured_face_for_demo[0])
                self.toggle_buttons(True)
                self.update_status("Face enhancement demo complete.", "blue")
                return

            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            cam_label.imgtk = imgtk
            cam_label.config(image=imgtk)
            
            # Use Tkinter hex color for info_label
            info_label.config(text="Looking for a face to capture for demo...", fg=rgb_to_hex((0, 0, 255))) # Red for scanning

            camera_window.after(10, update_demo_frame)

        camera_window.after(10, update_demo_frame)
        
        def on_demo_capture_close():
            nonlocal cap_demo # Declare cap_demo as nonlocal
            if cap_demo and cap_demo.isOpened():
                cap_demo.release()
            self.toggle_buttons(True)
            self.update_status("Face enhancement demo cancelled.", "orange")
            camera_window.destroy()

        camera_window.protocol("WM_DELETE_WINDOW", on_demo_capture_close)


    def _show_enhancement_results(self, face_img):
        enhancement_results_window = tk.Toplevel(self.master)
        enhancement_results_window.title("Face Enhancement Results")
        enhancement_results_window.geometry("700x700")

        tk.Label(enhancement_results_window, text="Original vs. Enhanced Face Images", font=("Helvetica", 16, "bold")).pack(pady=10)

        image_display_frame = tk.Frame(enhancement_results_window)
        image_display_frame.pack(pady=10)

        if len(face_img.shape) > 2:
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_img_gray = face_img

        enhanced_contrast = self.__face_enhancer.enhance_contrast(face_img_gray)
        reduced_noise = self.__face_enhancer.reduce_noise(face_img_gray)
        sharpened = self.__face_enhancer.sharpen(face_img_gray)

        self._create_image_frame(image_display_frame, face_img_gray, "Original", 0, 0)
        self._create_image_frame(image_display_frame, enhanced_contrast, "Contrast Enhanced", 0, 1)
        self._create_image_frame(image_display_frame, reduced_noise, "Noise Reduced", 1, 0)
        self._create_image_frame(image_display_frame, sharpened, "Sharpened", 1, 1)

        tk.Label(enhancement_results_window, text="\nThese techniques are applied automatically to detected faces before recognition to improve accuracy.", 
                 font=("Helvetica", 10), wraplength=680).pack(pady=10)
        
        close_btn = tk.Button(enhancement_results_window, text="Close", command=enhancement_results_window.destroy, font=("Helvetica", 12), bg="#607D8B", fg="white")
        close_btn.pack(pady=10)

    def _create_image_frame(self, parent_frame, image_array, label_text, row, column):
        frame = tk.LabelFrame(parent_frame, text=label_text, padx=5, pady=5)
        frame.grid(row=row, column=column, padx=10, pady=10)
        
        display_size = (150, 150)
        img_pil = Image.fromarray(image_array)
        img_pil = img_pil.resize(display_size, Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        label = tk.Label(frame, image=imgtk)
        label.image = imgtk
        label.pack()


    def toggle_buttons(self, enable):
        state = tk.NORMAL if enable else tk.DISABLED
        self.mark_attendance_btn.config(state=state)
        self.register_user_btn.config(state=state)
        self.view_attendance_btn.config(state=state)
        self.adjust_sensitivity_btn.config(state=state)
        self.demonstrate_enhancement_btn.config(state=state)

    def on_closing(self):
        if messagebox.askokcancel("Exit", "Do you want to exit the Facial Attendance System?"):
            self.running_camera = False
            if self.cap:
                self.cap.release()
            if self.camera_window and self.camera_window.winfo_exists():
                self.camera_window.destroy()
            self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemGUI(root)
    root.mainloop()