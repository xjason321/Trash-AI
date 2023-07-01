from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image

app = Flask(__name__, static_folder='static')

def preprocess_input(path):
    image = Image.open(path)
    new_image = image.resize((255, 255))
    new_image.save(path)

    im = Image.open(path)
    pxs = im.load()
    im_list = []
    for i in range(255):
        im_list.append([])
        for j in range(255):
            im_list[i].append(pxs[i, j])
    return (im_list[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'imageUpload' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['imageUpload']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file to a specific location
    file.save('static/uploads/' + file.filename)
    
    processed_input = preprocess_input(f'static/uploads/{file.filename}')
    
    # RUN NEURAL NETWORK ON PROCESSED INPUT.
    model = tf.keras.models.load_model('handwritten.model')
    
    # Make predictions and sort
    predictions = model.predict(processed_input)
    sorted_predictions = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

    mapping = {
        1: "battery", 2: "biological", 3: "brown-glass", 4: "cardboard", 5: "clothes", 6: "green-glass", 7:"metal", 8: "paper", 9: "plastic", 10: "shoes", 11: "trash", 12:"white-glass"
    }
    
    index1, likelihood1 = sorted_predictions[0]
    index2, likelihood2 = sorted_predictions[1]
    index3, likelihood3 = sorted_predictions[2]

    prediction = "plastic"

    sort = {
        "metal": "Metal",
        "brown-glass": "Glass",
        "white-glass": "Glass",
        "paper": "Paper",
        "plastic": "Plastic",
        "cardboard": "Cardboard",
        "green-glass": "Glass",
        "battery": "Battery",
        "biological": "Biological",
        "clothes": "Clothes",
        "shoes": "Shoes",
        "trash": "Trash"
    }

    instructions = {
        "Metal": "Metal waste includes items made of various types of metal, such as aluminum, steel, and tin. These can include beverage cans, food cans, metal packaging, small appliances, and other metal items. Recycling metal helps conserve natural resources and reduce the energy required for new metal production.",
        "Glass": "Glass waste includes bottles, jars, and other items made of glass. It is important to separate glass waste by color to facilitate the recycling process. Brown glass, white glass, and green glass can all be recycled. Please make sure to remove any caps or lids before recycling.",
        "Paper": "Paper waste includes newspapers, magazines, office paper, cardboard, and other similar items. To recycle paper, make sure it is clean and dry. Remove any plastic wrappers or other non-paper materials before recycling.",
        "Plastic": "Plastic waste includes various types of plastic containers, bottles, bags, and packaging materials. Recycling plastic helps reduce the consumption of new raw materials and saves energy. Remember to empty and rinse plastic containers before recycling.",
        "Cardboard": "Cardboard waste includes corrugated cardboard boxes and packaging materials. Flatten cardboard boxes to save space before recycling. Remove any tape or labels if possible.",
        "Battery": "Batteries are hazardous waste and should not be disposed of in regular trash. It is important to recycle batteries properly to prevent environmental contamination. Check with your local waste management facility for battery recycling options.",
        "Biological": "Biological waste includes organic materials such as food waste, plant trimmings, and other biodegradable substances. Composting is an environmentally friendly way to dispose of biological waste. Consider starting a compost bin or check if composting services are available in your area.",
        "Clothes": "Clothes that are in good condition can be donated to charitable organizations or sold second-hand. If clothes are damaged or worn out, they can be repurposed into cleaning rags or other textile recycling options.",
        "Shoes": "Shoes that are in good condition can be donated to charitable organizations or passed on to others who can use them. If the shoes are beyond repair or heavily worn out, consider recycling them through specialized shoe recycling programs.",
        "Trash": "Trash refers to waste that cannot be recycled or composted. This includes items such as broken glass, dirty diapers, and other non-recyclable/non-biodegradable materials. Properly disposing of trash helps maintain cleanliness and hygiene in your surroundings."
    }

    typeOfTrash = sort[prediction]
    instructionForDisposal = instructions[typeOfTrash]

    # Pass the filename to the template for display
    return render_template('index.html', filename=file.filename, prediction=prediction, type=typeOfTrash, instructions=instructionForDisposal)

# @app.route('/', methods=['POST'])
# def show_prediction():
#     model = tf.keras.models.load_model('handwritten.model')
    
#     # Make predictions and sort
#     predictions = model.predict(reshaped_array)
#     sorted_predictions = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

#     mapping = {
#         1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
#         9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P',
#         17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V',23: 'W', 24: 'X', 25: 'Y', 26: 'Z',
#         27: 'a', 28: 'b', 29: 'c', 30: 'd', 31: 'e', 32: 'f', 33: 'g', 34: 'h', 35: 'i',
#         36: 'j', 37: 'Unknown'
#     }
    
#     index1, likelihood1 = sorted_predictions[0]
#     index2, likelihood2 = sorted_predictions[1]
#     index3, likelihood3 = sorted_predictions[2]
#     index4, likelihood4 = sorted_predictions[3]
#     index5, likelihood5 = sorted_predictions[4]

#     m1 = f"Letter {mapping[index1]}: Likelihood {format(likelihood1 * 100, '.2f')}%" if likelihood1 > .10 else "The AI is struggling with this one. Try again!"
#     m2 = f"Letter {mapping[index2]}: Likelihood {format(likelihood2 * 100, '.2f')}%" if likelihood2 > .05 else " "
#     m3 = f"Letter {mapping[index3]}: Likelihood {format(likelihood3 * 100, '.2f')}%" if likelihood3 > .05 else " "
#     m4 = f"Letter {mapping[index4]}: Likelihood {format(likelihood4 * 100, '.2f')}%" if likelihood4 > .05 else " "
#     m5 = f"Letter {mapping[index5]}: Likelihood {format(likelihood5 * 100, '.2f')}%" if likelihood5 > .05 else " "

#     return render_template('index.html', grid=grid, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5)

if __name__ == '__main__':
    app.run()
