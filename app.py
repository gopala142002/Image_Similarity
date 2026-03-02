import os
import zipfile
import pandas as pd
import cv2
import torch
import lpips
import shutil 
import uuid

from flask import Flask, render_template, request, send_file
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net="alex").to(DEVICE)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_two_images(folder):
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(IMAGE_EXTENSIONS):
            images.append(os.path.join(folder, file))

    if len(images) != 2:
        return None, None

    return images[0], images[1]


def compute_metrics(img1, img2):

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    psnr_val = peak_signal_noise_ratio(img1, img2, data_range=255)

    ssim_val = structural_similarity(
        img1, img2,
        channel_axis=2,
        data_range=255
    )

    t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    t1 = t1 * 2 - 1
    t2 = t2 * 2 - 1

    t1 = t1.to(DEVICE)
    t2 = t2.to(DEVICE)

    lpips_val = lpips_model(t1, t2).item()

    return psnr_val, ssim_val, lpips_val

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_folder():

    file = request.files["folderzip"]

    if file.filename == "":
        return "No file uploaded!"

    zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(zip_path)

    unique_id = str(uuid.uuid4())
    extract_path = os.path.join(UPLOAD_FOLDER, unique_id)
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)  # delete zip after extraction

    results = []

    for student in sorted(os.listdir(extract_path)):
        student_path = os.path.join(extract_path, student)

        if not os.path.isdir(student_path):
            continue

        img1_path, img2_path = get_two_images(student_path)

        if img1_path is None:
            results.append({
                "Student": student,
                "PSNR": "N/A",
                "SSIM": "N/A",
                "LPIPS": "N/A"
            })
            continue

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        psnr, ssim, lp = compute_metrics(img1, img2)

        results.append({
            "Student": student,
            "PSNR": round(psnr, 4),
            "SSIM": round(ssim, 4),
            "LPIPS": round(lp, 4)
        })

    shutil.rmtree(extract_path, ignore_errors=True)

    df = pd.DataFrame(results)

    import io
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="Similarity_Report.csv"
    )
    
if __name__ == "__main__":
    app.run()
