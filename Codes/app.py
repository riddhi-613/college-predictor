import pickle
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open("model1.pkl", "rb"))

# Define the CSV file path
csv_file = "iit-and-nit-colleges-admission-criteria-version-2.csv"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')  

@app.route('/colleges')
def colleges():
    return render_template('Top Colleges.html')     

@app.route('/learn')
def learn():
    return render_template('Coding.html')     


@app.route('/faq')
def faq():
    return render_template('faq.html')              

@app.route('/predict', methods=['POST'])
def predict():
    # Define dictionaries for mapping categories, quota, pool, and institute types
    Category = {'0': 'General', '1': 'Other Backward Classes-Non Creamy Layer', '6': 'Scheduled Castes', 
                '8': 'Scheduled Tribes', '3': 'General & Persons with Disabilities', 
                '5': 'Other Backward Classes & Persons with Disabilities', 
                '7': 'Scheduled Castes & Persons with Disabilities', 
                '9': 'Scheduled Tribes & Persons with Disabilities', 
                '1': 'General & Economically Weaker Section', 
                '2': 'General & Economically Weaker Section & Persons with Disability'}
    
    Quota = {'0': 'All-India', '3': 'Home-State', '1': 'Andhra Pradesh', '2': 'Goa', '4': 'Jammu & Kashmir', '5': 'Ladakh'}
    Pool = {'0': 'Neutral', '1': 'Female Only'}
    Institute = {'0': 'IIT', '1': 'NIT'}

    # Get data from the form
    data = [x for x in request.form.values()]
    list1 = data.copy()

    # Map categorical inputs
    list1[2] = Category.get(list1[2])
    list1[3] = Quota.get(list1[3])
    list1[4] = Pool.get(list1[4])
    list1[5] = Institute.get(list1[5])

    # Prepare the data for prediction
    data.pop(0)
    data.pop(0)
    data.pop(7)
    data1 = [float(x) for x in data]
    final_output = np.array(data1).reshape(1, -1)
    output = model.predict(final_output)[0]

    # Append the predictions to the list
    list1.append(output[0])
    list1.append(output[1])
    list1.append(output[2])

    # Check if the CSV file exists, load it or create a new DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["year", "institute_type", "round_no", "quota", "pool", "institute_short",
                                   "program_name", "program_duration", "degree_short", "category",
                                   "opening_rank", "closing_rank", "is_preparatory", "predicted_college",
                                   "predicted_degree", "predicted_course"])

    # Ensure `list1` matches the DataFrame columns (14 elements for 14 columns)
    if len(list1) < len(df.columns):
        list1.extend([None] * (len(df.columns) - len(list1)))  # Add placeholders if list is too short
    elif len(list1) > len(df.columns):
        list1 = list1[:len(df.columns)]  # Trim list if it's too long

    # Append to the DataFrame and save to CSV
    new_row = pd.Series(list1, index=df.columns)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    # Display prediction results on the web page
    return render_template("home.html", prediction_text=f"College : {output[0]} , Degree : {output[1]} , Course : {output[2]}", 
                           prediction="Thank you, Hope this will match your requirement !!!")

if __name__ == '__main__':
    app.run(debug=True)
