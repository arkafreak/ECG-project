import streamlit as st
import pickle
import numpy as np

def load_model():
	with open("saved.pkl", 'rb') as file:
		data = pickle.load(file)
	return data

data = load_model()

regressor = data['model']
hr = data['HR']
age = data['age']

def show_predict_page():
	st.title("Heart Disease Prediciton")
	st.write("""### WE NEED SOME INFO FOR PREDICTION""")
	#HeartRate = {"50","51","52","53","54","55","56","57","58","59"
	#			,"60","61","62","63","64","65","66","67","68","69"
	#			,"70","71","72","73","74","75","76","77","78","79"
	#			,"80"}

	#Age = {"20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
	#		 "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
	#		  "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
	#		   "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
	#		    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
	#		     "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
	#		      "80"}
	Sex = {1,0}
	cp = {2, 3, 1}
	TrestBps = {120, 125}
	Cholestrol = {187, 188, 189}
	fbs = {1,0}
	RestEcg ={1,0}
	Thalach = {88,89,90,91, 92}
	Exang = {1 ,0}
	Oldpeak = {0,0.1, 0.2, 1.3}
	Slope = {0, 1, 2}
	Ca = {0, 1, 2}
	Thal = {0, 1, 2}
	Target = {1, 0}
	PR_interval = {114}
	RR_interval = {1.01, 1.02, 1.03, 1.04}




	Age = st.slider("Age", 20 , 80 , 20)
	Sex = st.selectbox("Gender", Sex)
	cp = st.selectbox("cp",cp)
	TrestBps = st.selectbox("TrestBps", TrestBps)
	Cholestrol = st.selectbox("Cholestrol", Cholestrol)
	fbs = st.selectbox("fbs", fbs)
	RestEcg = st.selectbox("RestEcg", RestEcg)
	Exang = st.selectbox("Exang", Exang)
	Oldpeak = st.selectbox("Oldpeak", Oldpeak)
	Slope = st.selectbox("Slope", Slope)
	Ca = st.selectbox("Ca" , Ca)
	Thal = st.selectbox("Thal", Thal)
	Target = st.selectbox("Target", Target)
	PR_interval = st.selectbox("PR_interval", PR_interval)
	RR_interval = st.selectbox("RR_interval", RR_interval)
	HeartRate = st.slider("HeartRate",40, 90 , 40)




	ok = st.button("Predict condition of heart")
	if ok:
		x = np.asarray([[Age,Sex, cp , TrestBps, Cholestrol,
		 fbs, RestEcg, Thalach, Exang, Oldpeak, Slope, Ca, Thal,
		 Target, PR_interval, RR_interval, HeartRate]])

		heartdata = regressor.predict(x)
		st.subheader(f"The condition of heart is: ${heartdata[0]}")
