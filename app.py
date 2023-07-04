from flask import Flask, render_template, request
from flask_cors import CORS
from tools.clarke_wright import cw_algorithm

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/cw', methods=['GET', 'POST'])
def cw():
    if request.method == 'POST':
        data = request.get_json()
        demand = data["demand"]
        
        distance_matrix = data["matrix"]
        truck_capacity = int(data["max_capacity"])
        max_distance = int(data["max_distance"])
        num_trucks = int(data["num_trucks"])
        result = {}
        output = cw_algorithm(distance_matrix=distance_matrix, demands=demand, vehicle_capacities=truck_capacity, max_distance=max_distance, num_vehicles=num_trucks, depot=0)
        if output : 
            result["output"] = output
            return result 
        else: 
            result["error"] = "Aucune solution possible en utilisant la méthode de Clarke & Wright!"
            return result
    else:
        return "Hola!"

if __name__ == '__main__':
    app.run(debug=True)
