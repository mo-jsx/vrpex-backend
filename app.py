from flask import Flask, render_template, request
from flask_cors import CORS
from tools.clarke_wright import cw_algorithm
from tools.gen import main_gen_algorithm


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

@app.route('/gen', methods=['GET', 'POST'])
def gen():
    if request.method == 'POST':
        data = request.get_json()
        demand = [(i+1, value) for i, value in enumerate(data["demand"])]
        
        distance_matrix = data["matrix"]
        truck_capacity = int(data["max_capacity"])
        max_distance = int(data["max_distance"])
        num_trucks = int(data["num_trucks"])
        nb_clients= int(len(demand))
        nb_iteration= int(data["nb_Iteration"])
        nb_selected_parents= int(data["nb_selected_parents"])
        
        population_size=int(data["populationSize"])
        result = {}
        # output = gen_algorithm(distance_matrix=distance_matrix, commands=demand, max_capacity_vehicule=truck_capacity, nb_vehicules=num_trucks, max_dist=max_distance, nb_clients=nb_clients, nb_iteration=nb_iteration, nb_selected_parents=nb_selected_parents, population_size=population_size)
        output = main_gen_algorithm(population_size=population_size, nb_iteration=nb_iteration, nb_clients=nb_clients, nb_selected_parents=nb_selected_parents, max_dist=max_distance, nb_vehicules=num_trucks, max_capacity_vehicule=truck_capacity, commands=demand, distance_matrix=distance_matrix)
        
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
