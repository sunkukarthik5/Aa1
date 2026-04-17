from flask import Flask, render_template, request, send_file, jsonify
import cv2
import pytesseract
import re
import os
import zipfile
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# ✅ IMPORTANT (Render/Linux fix)
pytesseract.pytesseract.tesseract_cmd = "tesseract"

UPLOAD = "uploads"
OUTPUT = "outputs"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# -----------------------------
# NAME LIST
# -----------------------------
NAMES_LIST = [
    "Pranathi Gundagani","Shashank Reddy Gantla","Sai Ravi Teja","SAI Lokesh Kalemula",
"Vaishnavi Suda","sai charan merugu","Pruthvi Teja","Ganesh Reddy","Sruthi Agarwal",
"Sai Kiran","Anvesh Reddy","Hanuk Potharaju","Sharik Shaik","Grishma Avula",
"Sanjana","Sai Teja Emmadishetty","Sreeja Anantha","Adharsh Malla","Sravan Kumar",
"Abhishek","Bhuvaneswar Reddy","Priyanka vasireddy","Renuka Gudavalli",
"Bhargav chintha","Prasad Kavi","Jashwanth naidu","Abhimanyu sinde",
"phaneendra kumar srungarapu","divya sri mullapudi","Ramu Machukuri","Akshaya",
"yeshwanth Thalluri","Nithish kumar","Rishitha kanugula","Aparna Golla",
"Vishnu Priya","Arpitha","Akhil addepalli","karthik gujjallapudi",
"Sowmya Mysore Srikanth","Sai kirreet reddy","Bhargavi Vemparala",
"Sai satwik Gampa","Shashidhar Reddy","Santhoshini goskula","Siva Koushik",
"yeshwanth chaganti","Dinesh Kumar","Mohammed Moinuddin Ansari",
"Vidya Malladi","Venugopal Reddy","Sathwik Reddy Chandiri",
"Nandakumar Reddy","Pranavi Reddy","Varun Reddy","Sai Teja Mosam",
"Sri divya","Pavan Sai Kandukuri","Satya Reddygari",
"Prudhvi Raj Rayapaneni","Naveen Vanga","Harsha Vardhan Sanka",
"Yeswanth Kumar","Gayathri Vigna Penmetsa","veera manda",
"Sirisha dewasoth","Naga Jagadeesh Krishna Kandula","Sai Kiran Katakam",
"Naga Raju","Tanusree Byram","krishna sri lakshmi","venkatesh kunchapu",
"Shireesha Reddy","Harshitha Sunkara","Murali Krishna","Gayathri Gali",
"Sathwik Gunda","uma maheswara reddy allu","sai saketh konakallaa",
"Venkata Sai Chedhalavada","Alekhya Shikaram","Kesava Chowdary",
"Trinadh Atmuri","nandini Bathini","sainath patel naini",
"Puneeth Cherukuri","Mahesh Mandalapu","Pavan Danthu","Ramu Moghili",
"Satvika","Srujani","Devendra Reddy","Harshitha Kapa","Sai Nithin",
"Satya Gudipudi","Manikata Mathadi","Sai krishna maddukuri",
"Daya sagar","sameeksha Thumma","Aravind Boddu","Nanditha Kothi",
"Jayanth","Tejaswini Yerra","Laxman Naragani","Mohitha Somepalli",
"Hemanth Narni","Bhuvana Chandrika","Rekha kosaraju","Hari sumanth",
"Venkata sharmi Pillella","Anand ram","Surya pramod vadapalli",
"Sai varun","vikram CH","Prasanna Soma","Ranadeep","Rohith Kumar",
"Venkatesh Edara","Sai vikram","Ravi Teja Reddy",
"Pranay Charan Kolasani","Risitha Gudipati","Rishikesh Reddy",
"Haritha Reddy","Srikanth Baikadi","Mounika Koduri","Dinesh kyanam",
"Vikas Yedavelly","Rohith Konduru","Sameera","Ganesh Edla",
"Tejaswi Alupula","Azmaan","Krishna Pudi","Susanth","SharanDeep",
"Venu Madhavi Kathi","Richitha Reddy","Hemanth Laxman",
"Preethu Prathyusha","Sravanthi","Azmaan Amin Hemraj",
"Alekha mandalapu","Aravind Reddy","sree kalyan Reddy",
"Uma Tamalapudi","Gnana Deepika","Bhavana pabboju",
"Avinash nadella","Nikhil kumar","Vamshi chaganti"
]

# -----------------------------
# REGEX
# -----------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
US_PHONE_RE = re.compile(r"(\+1\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
IND_PHONE_RE = re.compile(r"(\+91[\-\s]?)?[6-9]\d{9}")
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# BLUR FUNCTION
# -----------------------------
def strong_blur(img,x,y,w,h):
    if w<=0 or h<=0:
        return
    x=max(0,x); y=max(0,y)
    w=min(img.shape[1]-x,w)
    h=min(img.shape[0]-y,h)
    roi=img[y:y+h,x:x+w]
    blurred=cv2.GaussianBlur(roi,(99,99),40)
    img[y:y+h,x:x+w]=blurred

# -----------------------------
# FACE BLUR
# -----------------------------
face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

def blur_faces(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        strong_blur(img,x,y,w,h)
    return img

# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------
# UPLOAD + PROCESS
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "images" not in request.files:
            return jsonify({"error": "No images key found"}), 400

        files = request.files.getlist("images")

        memory_file = BytesIO()
        processed = 0

        with zipfile.ZipFile(memory_file,'w',zipfile.ZIP_DEFLATED) as zf:

            for file in files:

                if file.filename == '' or not allowed_file(file.filename):
                    continue

                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD, filename)
                file.save(path)

                img = cv2.imread(path)
                if img is None:
                    continue

                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                data = pytesseract.image_to_data(
                    gray,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'
                )

                for i, word in enumerate(data["text"]):
                    text = word.strip()
                    if not text:
                        continue

                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]

                    # EMAIL
                    email_match = EMAIL_RE.search(text)
                    if email_match:
                        strong_blur(img, x, y, w, h)
                        continue

                    # PHONE / URL
                    if (US_PHONE_RE.search(text) or
                        IND_PHONE_RE.search(text) or
                        URL_RE.search(text)):
                        strong_blur(img, x, y, w, h)
                        continue

                    # NAME CHECK
                    lower_text = text.lower()
                    for name in NAMES_LIST:
                        if any(part in lower_text for part in name.lower().split()):
                            strong_blur(img, x, y, w, h)
                            break

                img = blur_faces(img)
                img = cv2.resize(img, None, fx=0.5, fy=0.5)

                out_path = os.path.join(OUTPUT, "blurred_" + filename)
                cv2.imwrite(out_path, img)

                zf.write(out_path, os.path.basename(out_path))

                processed += 1

                os.remove(path)
                os.remove(out_path)

        memory_file.seek(0)

        if processed == 0:
            return jsonify({"error": "No valid images processed"}), 400

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='blurred_images.zip'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
