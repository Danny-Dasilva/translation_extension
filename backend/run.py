# 1. Import the necessary class
import os

# 2. Instantiate the text detection model
# By default, this will use the PP-OCRv5 server detection model.
# The model will be downloaded automatically on first use.
from paddleocr import PaddleOCR
model = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 3. Specify the path to your image file
# Replace this with the actual path to your manga JPG image
image_path = './debug_output/input_image_1760806814239.jpg'

# Create an output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found.")
    print("Please replace 'path/to/your/manga_page.jpg' with a valid file path.")
else:
    # 4. Run the prediction
    print(f"Running text detection on: {image_path}")
    try:
        output = model.predict(image_path)
        breakpoint()
        # 5. Process and display the results
        # The 'output' is a list containing results for each image processed.
        # Since we process one image, we access the first element.
        if output:
            result = output[0]  # Get first result object from list

            # Access the 'res' dictionary within json
            res_dict = result.json.get('res', {})

            # The coordinates are stored in 'dt_polys'
            # Each item is a polygon with four (x, y) points
            detected_polygons = res_dict['dt_polys']  # Access via json['res']['dt_polys']

            print(f"\nDetected {len(detected_polygons)} text boxes.")
            for i, box in enumerate(detected_polygons):
                # box may be list or numpy array
                if hasattr(box, 'tolist'):
                    print(f"  Box {i+1}: {box.tolist()}")
                else:
                    print(f"  Box {i+1}: {box}")

            # 6. Save the visualization and JSON output 
            # This saves an image with the detected boxes drawn on it
            result.save_to_img(save_path=os.path.join(output_dir, "detection_result.jpg"))
            
            # This saves the raw coordinates and scores to a JSON file
            result.save_to_json(save_path=os.path.join(output_dir, "detection_result.json"))
            
            print(f"\nVisualization saved to: {os.path.join(output_dir, 'detection_result.jpg')}")
            print(f"JSON output saved to: {os.path.join(output_dir, 'detection_result.json')}")
        else:
            print("No text was detected in the image.")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")