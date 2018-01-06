from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import titanic

app = Flask(__name__)
api = Api(app)

class Predictor(Resource):
    def post(self):
        pclass = int(request.json['pclass']);
        age = int(request.json['age']);
        male = int(request.json['male']);
        siblingCount = int(request.json['siblingCount']);
        parchCount = int(request.json['parchCount']);
        
        if (pclass == 1 or pclass == 2 or pclass == 3) and (age > 0 and age < 150) and (male == 0 or male == 1) and (siblingCount >= 0 and siblingCount < 100) and (parchCount >= 0 and parchCount < 100):  
            prediction = titanic.test(pclass, age, male, siblingCount, parchCount);
            return {'result': prediction} 
        else:
            return {'error' : 'Incorrect Input'}
            
titanic.train();
api.add_resource(Predictor, '/predictor') # Route_1


if __name__ == '__main__':
     app.run(port=5002)