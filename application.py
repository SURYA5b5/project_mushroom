from flask import Flask, render_template, request, redirect, url_for
from src.pipelines.predict_pipeline import CustomData, PredictionPipeline
import subprocess

application = Flask(__name__)
app = application

categorical_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                    'spore-print-color', 'population', 'habitat']

columns = categorical_cols

cap_shape_categories = ['x', 'f', 'k', 'b', 's', 'c']
cap_surface_categories = ['y', 's', 'f', 'g']
cap_color_categories = ['n', 'g', 'e', 'y', 'w', 'b', 'p', 'c', 'u', 'r']
bruises_categories = ['f', 't']
odor_categories = ['n', 'f', 'y', 's', 'a', 'l', 'p', 'c', 'm']
gill_attachment_categories = ['f', 'a']
gill_spacing_categories = ['c', 'w']
gill_size_categories = ['b', 'n']
gill_color_categories = ['b', 'p', 'w', 'n', 'g', 'h', 'u', 'k', 'e', 'y', 'o', 'r']
stalk_shape_categories = ['t', 'e']
stalk_root_categories = ['b', 'e', 'c', 'r']
stalk_surface_above_ring_categories = ['s', 'k', 'f', 'y']
stalk_surface_below_ring_categories = ['s', 'k', 'f', 'y']
stalk_color_above_ring_categories = ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
stalk_color_below_ring_categories = ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
veil_color_categories = ['w', 'n', 'o', 'y']
ring_number_categories = ['o', 't', 'n']
ring_type_categories = ['p', 'e', 'l', 'f', 'n']
spore_print_color_categories = ['w', 'n', 'k', 'h', 'r', 'u', 'o', 'y', 'b']
population_categories = ['v', 'y', 's', 'n', 'a', 'c']
habitat_categories = ['d', 'g', 'p', 'l', 'u', 'm', 'w']

drop_down_order = [cap_shape_categories, cap_surface_categories, cap_color_categories, bruises_categories,
                   odor_categories, gill_attachment_categories, gill_spacing_categories,
                   gill_size_categories, gill_color_categories, stalk_shape_categories,
                   stalk_root_categories, stalk_surface_above_ring_categories,
                   stalk_surface_below_ring_categories, stalk_color_above_ring_categories,
                   stalk_color_below_ring_categories, veil_color_categories,
                   ring_number_categories, ring_type_categories, spore_print_color_categories, population_categories,
                   habitat_categories]


@app.route("/")
def index():
    # Get the value of the training_complete parameter from the URL
    training_complete = request.args.get('training_complete', False)
    return render_template("home.html", col=columns, drop_down_order=drop_down_order,
                           categorical_cols=categorical_cols, enumerate=enumerate, training_complete=training_complete)


@app.route('/train', methods=['GET'])
def train_model():
    # Call the training pipeline script using subprocess
    subprocess.run(['python', 'src/pipelines/train_pipeline.py'])
    # Set the training_complete variable to True
    training_complete = True
    return redirect(url_for("home", training_complete=training_complete))


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    cap_shape = request.form['cap-shape']
    cap_surface = request.form['cap-surface']
    cap_color = request.form['cap-color']
    bruises = request.form['bruises']
    odor = request.form['odor']
    gill_attachment = request.form['gill-attachment']
    gill_spacing = request.form['gill-spacing']
    gill_size = request.form['gill-size']
    gill_color = request.form['gill-color']
    stalk_shape = request.form['stalk-shape']
    stalk_root = request.form['stalk-root']
    stalk_surface_above_ring = request.form['stalk-surface-above-ring']
    stalk_surface_below_ring = request.form['stalk-surface-below-ring']
    stalk_color_above_ring = request.form['stalk-color-above-ring']
    stalk_color_below_ring = request.form['stalk-color-below-ring']
    veil_color = request.form['veil-color']
    ring_number = request.form['ring-number']
    ring_type = request.form['ring-type']
    spore_print_color = request.form['spore-print-color']
    population = request.form['population']
    habitat = request.form['habitat']

    data = CustomData(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                      gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                      stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type,
                      spore_print_color, population, habitat)
    df = data.get_data_as_dataframe()
    model = PredictionPipeline()
    prediction = model.predict(df)
    if prediction[0] == "e":
        prediction = "Edible 👍"
    else:
        prediction = "Poisonous 👎"

    return render_template("result.html", predict=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
