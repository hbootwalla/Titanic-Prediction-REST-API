from flask import Flask, request
from flask_restful import Resource, Api
from titanic import Predictor 

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    
    def get(self):
        pclass = int(request.args.get('pclass'));
        age = int(request.args.get('age'));
        male = int(request.args.get('male'));
        siblingCount = int(request.args.get('siblingCount'));
        parchCount = int(request.args.get('parchCount'));
        
        if (pclass == 1 or pclass == 2 or pclass == 3) and (age > 0 and age < 150) and (male == 0 or male == 1) and (siblingCount >= 0 and siblingCount < 100) and (parchCount >= 0 and parchCount < 100):  
            
            prediction = predictor.test(pclass, age, male, siblingCount, parchCount);
            return {'result': prediction} 
        else:
            return {'error' : 'Incorrect Input'}
            
    def post(self):
        learning_rate = float(request.json['learning_rate']);
        epochs = int(request.json['epochs']);
        if learning_rate > 0.0:
            predictor.set_learning_rate(learning_rate);
        if epochs > 0:
            predictor.set_epochs(epochs);
        predictor.train();
        
      
predictor = Predictor();
predictor.train();      

api.add_resource(Prediction, '/prediction') 


if __name__ == '__main__':
     app.run(port=5002)