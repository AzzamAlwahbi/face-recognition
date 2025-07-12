import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Identify King Salman or Prince Mohammed")
        self.root.geometry("600x650")
        self.root.configure(bg='#f5f5f5')
        
        # Load model and labels
        self.model = load_model("keras_Model.h5", compile=False)
        self.class_names = self.load_labels("labels.txt")
        
        # Create UI elements
        self.create_widgets()
        
    def load_labels(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip().split(" ", 1)[1] for line in lines]
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="identify King Salman or Prince Mohammed", 
                        font=("Arial", 20, "bold"), bg='#f5f5f5', fg="#333")
        title.pack(pady=(20, 10))
        
        # Subtitle
        subtitle = tk.Label(self.root, text="Identify King Salman or Prince Mohammed", 
                           font=("Arial", 12), bg='#f5f5f5', fg="#666")
        subtitle.pack(pady=(0, 20))
        
        # Image display
        self.image_frame = tk.Frame(self.root, width=400, height=300, bg="#fff", 
                                   highlightbackground="#ddd", highlightthickness=1)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, text="Upload an image", 
                                  bg="#fff", fg="#999", font=("Arial", 12))
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg="#f5f5f5")
        button_frame.pack(pady=15)
        
        # Upload button
        upload_btn = tk.Button(button_frame, text="Upload Image", font=("Arial", 11),
                              command=self.upload_image, bg="#4a7abc", fg="white", 
                              padx=20, pady=8, relief="flat")
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Recognize button
        recognize_btn = tk.Button(button_frame, text="Recognize", font=("Arial", 11),
                                command=self.recognize_face, bg="#2c6e49", fg="white", 
                                padx=20, pady=8, relief="flat")
        recognize_btn.pack(side=tk.LEFT, padx=10)
        
        # Result frame
        result_frame = tk.Frame(self.root, bg="#f5f5f5")
        result_frame.pack(pady=20)
        
        self.result_label = tk.Label(result_frame, text="", font=("Arial", 14, "bold"),
                                    bg="#f5f5f5", fg="#2c6e49")
        self.result_label.pack()
        
        self.confidence_label = tk.Label(result_frame, text="", font=("Arial", 12),
                                        bg="#f5f5f5", fg="#555")
        self.confidence_label.pack()
        
        # Store image references
        self.img = None
        self.tk_img = None
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                self.img = Image.open(file_path)
                self.display_image(self.img)
                self.clear_results()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")
    
    def display_image(self, img):
        # Resize image to fit in frame
        img = img.copy()
        img.thumbnail((400, 300))
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img, text="")
        
    def clear_results(self):
        self.result_label.config(text="")
        self.confidence_label.config(text="")
    
    def recognize_face(self):
        if self.img is None:
            messagebox.showinfo("Info", "Please upload an image first.")
            return
        
        try:
            # Convert to OpenCV format
            cv_img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            processed_img = self.preprocess_image(cv_img)
            
            # Predict
            prediction = self.model.predict(processed_img)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]
            
            # Display result
            result = "King Salman" if class_name == "الملك سلمان" else "Prince Mohammed"
            self.result_label.config(text=f"Recognition Result: {result}")
            self.confidence_label.config(
                text=f"Confidence: {confidence_score * 100:.2f}%"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
    
    def preprocess_image(self, image):
        # Resize to model input size
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert to numpy array and reshape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        
        # Normalize the image
        image = (image / 127.5) - 1
        
        return image

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()