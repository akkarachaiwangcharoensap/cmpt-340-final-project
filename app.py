from flask import Flask, render_template, request, redirect, url_for, jsonify
from src.app.file_exceptions import FileUploadError
from src.app.upload import upload_file, remove_file
from src.app.conversion import convert_to_numpy, convert_to_png, image_to_base64
from src.app.metrics import get_metrics, metrics_to_bboxes, create_mask, create_dummy_mask, df_to_json
from src.preprocessing.preprocessing_pipeline import down_sample
import matplotlib.pyplot as plt
from src.models.unet_family.nnUNet.nnUNet_2D import nnunetv2_inference_api
import os

app = Flask(__name__)

# Ensure the tmp directory exists
os.makedirs('tmp', exist_ok=True)

# Main page route
@app.route('/')
def index():
    return render_template('index.html')

# Render about page.
@app.route('/acknowledgement')
def about():
    return render_template('acknowledgement.html')

# Demo page route (file upload page)
@app.route('/demo', methods=['POST', 'GET'])
def demo():
    if request.method == 'POST':
        file = request.files['file']
        try:
            shapes = upload_file(file)
            app.logger.info('A file is uploaded')
        except FileUploadError as e:
            app.logger.info('upload failed')
            return jsonify({"error": str(e)}), 500

        # Read image
        scan = convert_to_numpy(os.path.join('tmp', file.filename))
        # Construct input dictionary
        # prediction = create_dummy_mask(downsampled)
        # Call model
        if scan.shape == (512,512):
            downsampled = down_sample(scan)
            prediction = nnunetv2_inference_api({os.path.splitext(file.filename)[0]:downsampled})
        else:
            downsampled = scan
            prediction = nnunetv2_inference_api({os.path.splitext(file.filename)[0]:scan})

        # Read predicted mask (?binarize)
        mask = create_mask(prediction[os.path.splitext(file.filename)[0]])
        # call metrics function
        metrics = get_metrics(mask)

        # get scan and prediction into 8bit
        image = convert_to_png(downsampled)
        app.logger.info("Finished Prediction")
        # write images
        plt.imsave(
            os.path.join('tmp', 'image.png'),
            image, cmap='gray'
        )
        plt.imsave(
            os.path.join('tmp', 'mask.png'),
            mask, cmap='gray'
        )
        app.logger.info("Wrote Files")

        app.logger.info(metrics)

        data = df_to_json({
            "original_image": f"data:image/jpeg;base64,{str(image_to_base64(os.path.join('tmp', 'image.png')))}",
            "prediction": f"data:image/jpeg;base64,{str(image_to_base64(os.path.join('tmp', 'mask.png')))}",
            "table": metrics,
            "shapes": metrics_to_bboxes(metrics)
        })

        # To save local storage space, remove the file afterward.
        remove_file(file.filename)
        return jsonify(data)

    return render_template('demo.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
