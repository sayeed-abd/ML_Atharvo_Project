from flask import Flask, request, jsonify,render_template
import pandas as pd
from processing.pre_processing import pre_process_data
import pickle

app = Flask(__name__)
patient_model1 = pickle.load(open('model.pkl', 'rb'))
patient_model = pickle.load(open('lr_model.pkl', 'rb'))

#Defines a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


#New
@app.route('/upload', methods=['POST'])
def upload_csv():    
    csv_file = request.files['file']
    if csv_file:
        data = pd.read_csv(csv_file)
        X_data = pre_process_data(data)            
        response_final_lr = patient_model.predict(X_data)
        response_final_rfc = patient_model1.predict(X_data)
        
        # Map the predicted tags to their corresponding labels
        labels = ['Angioplasty', 'Coronary Artery Bypass Graft (CABG)', 'Lifestyle Changes', 'Medication']
        tags = [0, 1, 2, 4]
        
        p_status_lr = labels[tags.index(response_final_lr[0])] if response_final_lr[0] in tags else 'Unknown'
        p_status_rfc = labels[tags.index(response_final_rfc[0])] if response_final_rfc[0] in tags else 'Unknown'
        
        # Example of using multiple conditions
        if response_final_lr[0] in tags:
            if response_final_rfc[0] in tags:
                return render_template('input.html', 
                                       prediction_text_lr='Prediction Result from Logistic Regression Algorithm = {}'.format(p_status_lr), 
                                       prediction_text_rfc='Prediction Result from Random Forest Classification = {}'.format(p_status_rfc))
            else:
                return render_template('input.html', 
                                       prediction_text_lr='Logistic Regression Prediction = {}'.format(p_status_lr), 
                                       prediction_text_rfc='Random Forest Prediction = Unknown')
        else:
            return render_template('input.html', 
                                   prediction_text_lr='Logistic Regression Prediction = Unknown', 
                                   prediction_text_rfc='Random Forest Prediction = {}'.format(p_status_rfc))
    else:
        return jsonify({'error': 'No file uploaded'})


# Route for uploading CSV file
# @app.route('/upload', methods=['POST'])
# def upload_csv():    
#     csv_file = request.files['file']
#     if csv_file:
#         data = pd.read_csv(csv_file)
#         X_data = pre_process_data(data)            
#         response_final_lr = patient_model.predict(X_data)
#         response_final_rfc = patient_model1.predict(X_data)
#         # Map the predicted tags to their corresponding labels
#         labels = ['Angioplasty', 'Coronary Artery Bypass Graft (CABG)', 'Lifestyle Changes', 'Medication']
#         tags = [0, 1, 2, 4]
    
#         p_status_lr = labels[tags.index(response_final_lr[0])] if response_final_lr[0] in tags else 'Unknown'
#         p_status_rfc = labels[tags.index(response_final_rfc[0])] if response_final_rfc[0] in tags else 'Unknown'
#         return render_template('input.html', 
#                                 prediction_text_lr='Prediction Result from Logistic Regression Algorithm = {}'.format(p_status_lr), 
#                                 prediction_text_rfc='Prediction Result from Random Forest Classification = {}'.format(p_status_rfc))
#     # else:
#     #     return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
