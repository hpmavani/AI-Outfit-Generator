import base64
from flask import Flask, render_template, request, send_file
from webapp.Modules.image_loader import ImageLoader
from webapp.Modules.model import ModelHandler

app = Flask(__name__)
im = ImageLoader("data/test_no_dup.json")
ml = ModelHandler()

@app.route("/")
def index(): 
    return render_template("index.html")

@app.route("/api/images")
def images_api(): 
    offset = request.args.get('offset')
    limit = request.args.get('limit')
    image_data = im.get_all_images(limit, offset)
    image_data_json = []
    for item in image_data: 
        item_id = item[0]
        image_blob = item[2]
        image_base64 = base64.b64encode(image_blob).decode("utf-8") 
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        item_name = item[3]
        image_data_json.append({"item_id": item_id, "item_name": item_name, "image_url": image_data_url})
    return image_data_json

@app.route("/api/predict", methods=["POST"])
def predict(): 
    data = request.get_json()
    items = data['items']
    output = ml.predict(items)
    res = im.similarity_search(output)
    item_id = res[0]
    image_blob = res[2]
    item_name = res[3]
    
    image_base64 = base64.b64encode(image_blob).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"
    return {"item_id": item_id, "item_name": item_name, "image_url": image_data_url}

@app.route("/test")
def test(): 
    return render_template("image.html")

if __name__ == "__main__": 
    app.run(debug=True) 
    