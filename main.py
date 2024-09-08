import os
import cv2
import numpy as np
from flask import Flask, request, render_template_string
from google.cloud import vision
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'peak-tract-424210-t2-5a5aee9984a6.json'

app = Flask(__name__)

def upload_page():
    html_content = '<!doctype html><html>'
    html_content += '<head><title>Process your Image</title></head>'
    html_content += '<body>'
    html_content += '<h1 style="text-align: center;">Extract Text and Visual Elements from your IMAGE</h1>'
    html_content += '<h2>Upload an Image</h2>'
    html_content += '<form action="/" method="POST" enctype="multipart/form-data">'
    html_content += '<input type="file" name="image" accept="image/*" required>'
    html_content += '<button type="submit">Upload</button>'
    html_content += '</form>'
    html_content += '</body></html>'
    return html_content

def create_html_structure(text, visual_elements, visual_elements2):
    html_content = '<!doctype html><html>'
    html_content += '<head><title>Process your Image</title></head>'
    html_content += '<style>img { border: 2px solid red; max-width: 200px; max-height: 200px;}</style>'
    html_content += '<body>'
    html_content += '<h1 style="text-align: center;">Extract Text and Visual Elements from your IMAGE</h1>'
    html_content += '<h2>Upload an Image</h2>'
    html_content += '<form action="/" method="POST" enctype="multipart/form-data">'
    html_content += '<input type="file" name="image" accept="image/*" required>'
    html_content += '<button type="submit">Upload</button>'
    html_content += '</form>'
    html_content += '</br></br>'
    html_content += '<h2>Extracted content of the uploaded Image</h2>'
    html_content += '<h3> Extracted Text: </h3>'
    html_content += f'<p>{text}</p>'
    html_content += '<h3> Extracted Visual Elements without text: </h3>'
    for element in visual_elements:
        html_content += f'&nbsp;<img src="{element}" alt="Visual Element">&nbsp;&nbsp;&nbsp;'
    html_content += '</br>'
    html_content += '<h3> Extracted Visual Elements with text: </h3>'
    for element in visual_elements2:
        html_content += f'&nbsp;<img src="{element}" alt="Visual Element">&nbsp;&nbsp;&nbsp;'
    html_content += '</body></html>'
    return html_content


def extract_text_and_mask_image(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(response.error.message)
    extracted_text = texts[0].description if texts else ''
    return extracted_text, texts

def fill_bounding_boxes(image, texts):
    bounding_boxes = []
    for text in texts[1:]:
        box = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        bounding_boxes.append(box)

        x_coords = [vertex[0] for vertex in box]
        y_coords = [vertex[1] for vertex in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)        
        
        margin = 1
        x_coord = x_min - margin
        if x_coord <= 0: x_coord = x_max + margin
        y_coord = y_min - margin
        if y_coord <= 0: y_coord = y_max + margin
        color = image[y_coord, x_coord]
        
        color = tuple(map(int, color))
        cv2.fillPoly(image, [np.array(box, np.int32)], color)
    return image, bounding_boxes

def create_mask(image_shape, excluded_boxes):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for box in excluded_boxes:
        x_tl, y_tl = box[0]
        x_br, y_br = box[2]
        mask[y_tl:y_br, x_tl:x_br] = 255
    return mask

def segment_visual_elements(image_path, output_dir, texts):
    image = cv2.imread(image_path)
    image, _ = fill_bounding_boxes(image, texts)
    # cv2.imwrite(f"{output_dir}/fill_bounding_box.png", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(f"{output_dir}/binary_simple.png", binary)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"{output_dir}/binary_adaptive.png", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visual_elements = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 40 < w < image.shape[1]-50 and 40 < h < image.shape[0]-50 and w/h < 7 and h/w < 7:
            roi = image[y:y+h, x:x+w]   
            element_path = os.path.join(output_dir, f'element_fill_{i}.png')
            cv2.imwrite(element_path, roi)
            visual_elements.append(element_path)

    return visual_elements

def segment_visual_elements2(image_path, output_dir, texts):
    image = cv2.imread(image_path)
    _, bounding_boxes = fill_bounding_boxes(image, texts)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    mask = create_mask(image.shape[:2], bounding_boxes)
    inverted_mask = cv2.bitwise_not(mask)
    combined_mask = cv2.bitwise_and(binary, inverted_mask)
    # cv2.imwrite(f"{output_dir}/combined_mask.png", combined_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visual_elements = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 40 < w < image.shape[1]-50 and 40 < h < image.shape[0]-50 and w/h < 7 and h/w < 7:
            roi = image[y:y+h, x:x+w]
            element_path = os.path.join(output_dir, f'element_box_{i}.png')
            cv2.imwrite(element_path, roi)
            visual_elements.append(element_path)

    return visual_elements

@app.route('/', methods=['GET', 'POST'])
def upload_image(): 
    if request.method == 'POST'  and 'image' in request.files:
        image = request.files['image']
        os.makedirs('static', exist_ok=True)
        image_path = f'static/{image.filename}'
        image.save(image_path)

        extracted_text, texts = extract_text_and_mask_image(image_path)
        visual_elements = segment_visual_elements(image_path, 'static', texts)
        visual_elements2 = segment_visual_elements2(image_path, 'static', texts)
        print("visual_elements: ", visual_elements)
        print("visual_elements2: ", visual_elements2)
        html_content = create_html_structure(extracted_text, visual_elements, visual_elements2)
        return render_template_string(html_content)
    
    html_content = upload_page()
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True)
