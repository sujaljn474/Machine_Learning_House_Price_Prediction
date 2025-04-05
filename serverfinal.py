from flask import Flask, request, render_template
import pickle

# load the model
with open("rf.pkl", 'rb') as file:
    model = pickle.load(file)

with open("label_encoder.pkl",'rb') as file1:
    encoder=pickle.load(file1)

# create the server
app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return render_template("indexfinal.html")


@app.route("/predict", methods=["POST"])
def predict():
    # get input from user
    print(request.form)
    size = float(request.form.get("size"))
    society = request.form.get("society")
    location = request.form.get("location")
    totalSquareFeet = float(request.form.get("totalSquareFeet"))
    availableBy = request.form.get("availableBy")
    status = request.form.get("status")
    age = float(request.form.get("age"))

    # size = float(request.form.get("size"))
    # society = float(request.form.get("society"))
    # location = float(request.form.get("location"))
    # totalSquareFeet = float(request.form.get("totalSquareFeet"))
    # availableBy = float(request.form.get("availableBy"))
    # status = float(request.form.get("status"))
    # age = float(request.form.get("age"))

    society=encoder.fit_transform([society])
    so=society[0]
    location = encoder.fit_transform([location])
    lo=location[0]
    availableBy = encoder.fit_transform([availableBy])
    av=availableBy[0]
    status = encoder.fit_transform([status])
    st=status[0]
    predictions = model.predict([[size, so,lo,totalSquareFeet,av,st]])
    # return f"price = {predictions[0]}L"
    if len(str(predictions[0]).split(".")[0]) >= 3:
        result = f"{predictions[0]/100:.2f} Cr"
    else:
        result = f"{predictions[0]:.2f} L"
    return render_template('resultfinal.html', result=result)


# start the server
app.run(host="0.0.0.0", port=8000, debug=True)
