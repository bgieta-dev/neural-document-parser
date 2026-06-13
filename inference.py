import numpy as np
import onnxruntime as ort
from image_processor import ImageProcessor


class DigitPredictor:
    def __init__(self, model_path="digit_cnn.onnx"):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, canvas):
        # < 0, 1>
        img_data = canvas.astype(np.float32) / 255.0
        # <-1, 1>
        img_data = (img_data - 0.5) / 0.5
        # [BatchSize, Channels, Height, Width]
        img_data = img_data[np.newaxis, np.newaxis, :, :]
        # Tensor -> ONNX model
        outputs = self.session.run(None, {self.input_name: img_data})
        # Highest probability
        prediction = np.argmax(outputs[0])
        return prediction + 5


def main():
    pdf_path = "raw_data/3.pdf"
    processor = ImageProcessor(pdf_path)
    predictor = DigitPredictor("digit_cnn.onnx")

    cell_images = processor.get_cell_images()

    final_results = {}
    for table_id, images in cell_images.items():
        table_results = []
        for canvas in images:
            number = predictor.predict(canvas)
            table_results.append(int(number))
        final_results[table_id] = table_results


if __name__ == "__main__":
    main()
