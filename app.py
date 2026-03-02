import os
import zipfile
import pandas as pd
import cv2
import torch
import lpips
import shutil
import uuid
import io

from flask import Flask, render_template, request, send_file
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Limit upload size (50MB)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = "cpu"
lpips_model = None

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_lpips_model():
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net="alex").to("cpu")
    return lpips_model


def compute_metrics(img1, img2):

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    psnr_val = peak_signal_noise_ratio(img1, img2, data_range=255)

    ssim_val = structural_similarity(
        img1,
        img2,
        channel_axis=2,
        data_range=255
    )

    # Resize only for LPIPS (reduce memory usage)
    img1_small = cv2.resize(img1, (256, 256))
    img2_small = cv2.resize(img2, (256, 256))

    t1 = torch.tensor(img1_small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.tensor(img2_small).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    t1 = (t1 * 2 - 1)
    t2 = (t2 * 2 - 1)

    model = get_lpips_model()

    with torch.no_grad():
        lpips_val = model(t1, t2).item()

    return psnr_val, ssim_val, lpips_val


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_folder():

    file = request.files.get("folderzip")

    if not file or file.filename == "":
        return "No file uploaded!", 400

    # Secure filename
    filename = secure_filename(file.filename)
    zip_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(zip_path)

    unique_id = str(uuid.uuid4())
    extract_path = os.path.join(UPLOAD_FOLDER, unique_id)
    os.makedirs(extract_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        os.remove(zip_path)
        shutil.rmtree(extract_path, ignore_errors=True)
        return "Invalid ZIP file!", 400

    os.remove(zip_path)

    results = []

    for root, _, files in os.walk(extract_path):

        image_files = [
            f for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if len(image_files) >= 2:

            student_name = os.path.basename(root)

            img1_path = os.path.join(root, image_files[0])
            img2_path = os.path.join(root, image_files[1])

            img1 = load_image(img1_path)
            img2 = load_image(img2_path)

            if img1 is None or img2 is None:
                continue

            try:
                psnr, ssim, lp = compute_metrics(img1, img2)

                results.append({
                    "Student": student_name,
                    "PSNR": round(psnr, 4),
                    "SSIM": round(ssim, 4),
                    "LPIPS": round(lp, 4)
                })

            except Exception as e:
                print("Error:", e)

    shutil.rmtree(extract_path, ignore_errors=True)

    if not results:
        return "No valid image pairs found.", 400

    df = pd.DataFrame(results)

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="Similarity_Report.csv"
    )

    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)