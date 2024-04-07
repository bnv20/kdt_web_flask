from flask import Flask, render_template, request, redirect
import torch
from PIL import Image
import io
import os
import json
from ast import literal_eval
import argparse
import collections

# 폴더 경로 설정
BEFORE_FOLDER = "static/bef"
AFTER_FOLDER = "static/aft"


# 폴더 내의 모든 파일을 삭제하는 함수
def delete_all_files(folder_path):
    if os.path.exists(folder_path):
        for file in os.scandir(folder_path):
            os.remove(file.path)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # before 및 after 폴더 내 파일 삭제
        delete_all_files(BEFORE_FOLDER)
        delete_all_files(AFTER_FOLDER)

        if "file" not in request.files:
            return redirect(request.url)
        files = request.files.getlist("file")
        if not files:
            return redirect(request.url)

        resultlist, pf, file_names = [], [], []

        for uploaded_file in files:
            filename = uploaded_file.filename
            img_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img.save(f"{BEFORE_FOLDER}/{filename}", format="JPEG")

            # 이미지 처리 로직 (여기서는 YOLOv5 모델 사용)
            results = model(img, size=640)
            rendered_images = results.render()
            results_list = results.pandas().xyxy[0].to_json(orient="records")
            results_list = literal_eval(results_list)
            classes_list = [item["name"] for item in results_list]
            result_counter = collections.Counter(classes_list)

            detected_image_path = f"{AFTER_FOLDER}/{filename}"
            Image.fromarray(rendered_images[0]).save(detected_image_path, format="JPEG")
            file_names.append(filename)

            resultlist.append(json.dumps(dict(result_counter)))
            pf.append("PASS" if len(results_list) == 0 else "FAIL")

        firstimage = file_names[0] if file_names else None

        return render_template(
            "imageshow.html",
            files=file_names,
            resultlist=resultlist,
            pf=pf,
            firstimage=firstimage,
            enumerate=enumerate,
            len=len,
        )
    return render_template("detect.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # 모델 로드
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="best.pt", autoshape=True
    )
    model.eval()

    app.run(host="0.0.0.0", debug=True, port=args.port, threaded=True)
