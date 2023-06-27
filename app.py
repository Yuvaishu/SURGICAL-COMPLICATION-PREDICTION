from flask import Flask,render_template,url_for,request
import numpy as np
import tensorflow
import joblib

model =tensorflow.keras.models.load_model(r'C:\Miniproject\model\my_model_train_71_test_23.h5')
loaded_encoder = joblib.load(r'C:\Miniproject\model\Labelencoder.pkl')

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def yuva():
	if request.method=="POST":
		diabeties = request.form['diabeties']
		hypertension = request.form['hypertension']
		glaucoma = request.form['glaucoma']
		diabeticrectinopathy = request.form['diabeticrectinopathy']
		highmyopia = request.form['highmyopia']
		sicklecellanemia = request.form['sicklecellanemia']
		aids = request.form['aids']
		input = np.array([diabeties,hypertension,glaucoma,diabeticrectinopathy,highmyopia,sicklecellanemia,aids])
		result = model.predict(input.astype(int).reshape(1,-1))
		input = np.argmax(result, axis=1)
		result = loaded_encoder.inverse_transform(input)
		print(result)
		return render_template('mini.html',results=result[0])
	return render_template("mini.html",results="Predictions here!") 
if __name__ == '__main__':
	app.run(debug=True)