import random


class Truck:
    def __init__(self, capacity, max_distance):
        self.capacity = capacity
        self.commands = []
        self.weight = 0
        self.max_distance = max_distance
        self.current_distance = 0

    def can_add_command(self, command_id, command_weight, distance_matrix):
        if not self.commands:
            # Calculate distance from depot to the first command
            depot_to_first_command_distance = distance_matrix[0][command_id]
            # Check if the weight and distance constraints are satisfied
            return (
                    command_weight <= self.capacity
                    and depot_to_first_command_distance + self.current_distance + distance_matrix[command_id][
                        0] <= self.max_distance
            )
        else:
            # Calculate distance from the last command to the new command
            last_command_id = self.commands[-1]
            last_command_to_new_command_distance = distance_matrix[last_command_id][command_id]
            # Check if the weight and distance constraints are satisfied
            return (
                    self.weight + command_weight <= self.capacity
                    and self.current_distance + last_command_to_new_command_distance + distance_matrix[command_id][
                        0] <= self.max_distance
            )

    def add_command(self, command_id, command_weight, distance_matrix):
        if self.commands:
            # Calculate distance from the last command to the new command
            last_command_id = self.commands[-1]
            last_command_to_new_command_distance = distance_matrix[last_command_id][command_id]
            self.current_distance += last_command_to_new_command_distance
        else:
            # Calculate distance from depot to the first command
            depot_to_first_command_distance = distance_matrix[0][command_id]
            self.current_distance += depot_to_first_command_distance

        self.commands.append(command_id)
        self.weight += command_weight

    def exceeds_max_distance(self, distance_matrix):
        if self.commands:
            # Calculate distance from the last command to the depot
            last_command_id = self.commands[-1]
            last_command_to_depot_distance = distance_matrix[last_command_id][0]
            return self.current_distance + last_command_to_depot_distance > self.max_distance
        return False


# =================================================================================================================

class CommandAllocator:
    import random
    def __init__(self, commands, truck_capacity, distance_matrix, max_distance, num_trucks):
        self.commands = random.sample(commands, len(commands))  # Mélange aléatoire de la liste des commandes
        self.truck_capacity = truck_capacity
        self.distance_matrix = distance_matrix
        self.max_distance = max_distance
        self.num_trucks = num_trucks
        self.trucks = []
        self.unassigned_commands = []

        # init la solution générée
        self.delivery_solution = []

    def assign_trucks(self):
        for command_id, command_weight in self.commands:
            assigned = False
            for truck in self.trucks:
                if (
                        truck.can_add_command(command_id, command_weight, self.distance_matrix)
                        and not truck.exceeds_max_distance(self.distance_matrix)
                ):
                    truck.add_command(command_id, command_weight, self.distance_matrix)
                    assigned = True
                    break

            if not assigned:
                if len(self.trucks) < self.num_trucks:
                    new_truck = Truck(self.truck_capacity, self.max_distance)
                    new_truck.add_command(command_id, command_weight, self.distance_matrix)
                    self.trucks.append(new_truck)
                else:
                    self.unassigned_commands.append((command_id, command_weight))

    def print_truck_assignments(self):

        print("Number of trucks used:", len(self.trucks))
        print("Assignments of commands to trucks:")
        for i, truck in enumerate(self.trucks):
            if truck.commands:
                delivery = [0]
                tour = f"Truck {i + 1}: Dépôt -> "
                for command_id in truck.commands:
                    tour += f"Client {command_id} -> "
                    delivery.append(command_id)  # ajouter le command_id au sous-tournée

                tour += "Dépôt"

                # ajouter zero a la fin [dépot]
                delivery.append(0)

                self.delivery_solution.append(delivery)
                print(tour)
            print(f'delivery : {delivery}')

    def print_unassigned_commands(self):
        if self.unassigned_commands:
            print("Unassigned commands due to insufficient trucks:")
            for command_id, _ in self.unassigned_commands:
                print(f"Command {command_id}")

    def assign_trucks_and_return_results(self):
        self.assign_trucks()
        self.print_truck_assignments()
        self.print_unassigned_commands()
        return len(self.trucks), self.trucks


# =================================================================================================================

def gen_distance_matrix(nombre_clients):
    import numpy as np
    matrix = np.random.rand(nombre_clients+1, nombre_clients+1)*19 + 1
    matrix = np.triu(matrix) + np.triu(matrix, 1).T
    np.fill_diagonal(matrix, 0)
    return matrix


# =================================================================================================================
def read_dist_matrix_from_file(file, delimiter):
    with open(file, 'r') as f:
        lines = f.readlines()
        matrix = [[int(i) for i in line.strip().replace('[[','').replace('],','').replace('[','').replace(']]','').split(delimiter)] for line in lines if line.strip()]
    return matrix


# =================================================================================================================

# FITNESS FUNCTION
def fitness(l, matrix):
    d = 0
    for i in range(len(l) - 1):
        d += matrix[l[i]][l[i + 1]]
    return d


# =================================================================================================================

# CREATE TOUR GEANT FUNCTION
# input : list des sous-tours
def creer_tour_geant(sous_tours):
    new_list = [0]
    new_list.extend([elem for sublist in sous_tours for elem in sublist[1:-1]])
    new_list.append(0)
    return new_list

# =================================================================================================================

# fonction d'initialisation pardéfaut des données
def init_data(number_clients=5, truck_capacity=50, max_distance=70, num_trucks=5):
    commands = [(i, random.randint(2, 25)) for i in range(1, number_clients + 1)]
    distance_matrix = gen_distance_matrix(number_clients)
    return number_clients, commands, distance_matrix, truck_capacity, max_distance, num_trucks


# =================================================================================================================

def generate_2_random_routes(tours_geants=[]):
    import random
    tours = tours_geants.copy()
    if len(tours) > 1:
        rand1 = tours.pop(random.randint(0, len(tours)-1))
        rand2 = tours.pop(random.randint(0, len(tours)-1))
        return rand1, rand2, tours

# =================================================================================================================

def get_command(client, commands):
    return (client,dict(commands).get(client))

# =================================================================================================================

def sub_children(child):
    tmp = child[1:-1]

    # utiliser pour obtenir les éléments qui sont pas ajoutés a une sous-route
    sols = []

    # liste des sous tour obtenus de chq enfant en entrée
    results = []

    while len(tmp) != 0:

        sol = [0]
        for i in tmp:
            f = fitness(sol + [i] + [0], distance_matrix)
            cmd = sum([get_command(i, commands)[1] for i in sol[1::] + [i] if i != 0])

            print(f'{sol + [i] + [0]} : {f},  cmd : {cmd}')
            if f <= max_distance and cmd <= truck_capacity:
                sol.append(i)
        # Ajouter zero a la fin de chq sous tour
        sol.append(0)

        # mettre a jour la liste SOLS
        sols.extend(sol)

        # ajouter le résultat a la liste des résultats finaux
        results.append(sol)
        print(f'SOL : {sol} ')

        # mettre a jour la liste sur laquelle on travaille, supprimer les éléments qui sont déja sélectionnés
        tmp = [j for j in tmp if j not in sols]
        print(f'tmp : {tmp} ')

    return results


# =================================================================================================================

def generate_init_population(population_size):
    print(
        '*********************************************************************************************************************')
    print(
        '----------------------------------------GENERATION DE POPULATION INITIALE---------------------------------------- ...')
    print(
        '*********************************************************************************************************************')

    # LISTE DES TOURS GEANTS GENERES
    TOURS_GEANTS = []

    # GENERER 10 SOLUTIONS
    for i in range(population_size):

        # Appel de la  class CommandAllocator
        allocator = CommandAllocator(commands, truck_capacity, distance_matrix, max_distance, num_trucks)
        num_trucks_used, assigned_trucks = allocator.assign_trucks_and_return_results()

        solution = allocator.delivery_solution
        print(
            '=====================================================================================================================')
        print(f'SOLUTION N° {i + 1} : {solution}')
        print(f'TOUR GEANT N° {i + 1} : {creer_tour_geant(solution)}')
        print(
            '=====================================================================================================================')

        solution_fitness_list = []
        solution_fitness = 0

        # CALCUL DE DISTANCE DE CHQ SOUS-SOULTION
        for sol in solution:
            # appeler la fonction fitness qui calcule la distance d'un sous-tour
            f = fitness(sol, distance_matrix)
            solution_fitness_list.append(f)
            solution_fitness += f

        TOURS_GEANTS.append((creer_tour_geant(solution), solution_fitness))
    return TOURS_GEANTS


# =================================================================================================================

# prend en entrée les tours géants
def selection(tours_geants=[], nb_parents=len([]) // 2):
    print(
        '*********************************************************************************************************************')
    print(
        '----------------------------------------     OPERATION DE SELECTION      ---------------------------------------- ...')
    print(
        '*********************************************************************************************************************')

    input_routes = tours_geants.copy()

    # les tournées qui seront sélectionnées après tournoi
    tournees_gagnantes = []
    # print(f'======*================*=======')
    # print(f'taille initiale de input_routes : {len(input_routes)}')
    # print(f'======*================*=======')

    for i in range(nb_parents):
        # print(f'iteration  : {i+1}')

        # générer tous tours aléatoires et les supprimer de la liste des tours
        t1, t2, input_routes = generate_2_random_routes(input_routes)
        print(f'random selected routes  : {t1} and  {t2}')
        # print(f'taille de input_routes : {len(input_routes)}')

        # ajouter a la liste des selectionnés le min fitness des deux tours sélectionnés
        tournees_gagnantes.append(min([t1, t2], key=lambda x: x[1]))

        # rendre l'élément non sélectionné à la liste
        input_routes.append(max([t1, t2], key=lambda x: x[1]))

        print(f'selected route is :================> {min([t1, t2], key=lambda x: x[1])}')

        # print(f'>>>>>>>>> fin iteration <<<<<<<<<<<< taille de input_routes : {len(input_routes)}')

    return tournees_gagnantes


# =================================================================================================================

def croisement(selected_routes_input=[]):
    import random

    print('********************************************************************************************************************')
    print('------------------------------------------     OPERATION DE CROISEMENT      ------------------------------------------')
    print('********************************************************************************************************************')

    routes_selected = selected_routes_input.copy()

    parents = selected_routes_input.copy()
    enfants = []

    a1 = 0

    while len(routes_selected) != 0:

        # Générer un nombre aléatoire r1 entre 0 et 1 🙂
        r1 = random.random()
        print(f'Nombre aléatoire généré r1 = {r1}')

        # Si r1 <= 0.9 Alors
        if r1 <= 0.98:
            t11, t22, routes_selected = generate_2_random_routes(routes_selected)
            t1, t2 = t11[0][1:-1], t22[0][1:-1]
            print(f'PARENTS  : {t1}, {t2}')

            p1 = random.randint(1, number_clients - 2)
            p2 = random.randint(p1 + 1, number_clients - 1)

            A = t1[0:p1]
            APrime = t2[0:p1]
            B = t1[p1:p2]
            BPrime = t2[p1:p2]
            C = t1[p2::]
            CPrime = t2[p2::]

            matrice_correspendance_Parent_1 = {
                k: B[BPrime.index(k)] for k in BPrime
            }

            matrice_correspendance_Parent_2 = {
                k: BPrime[B.index(k)] for k in B
            }

            AX = ['x' for _ in range(len(A))]
            CX = ['x' for _ in range(len(C))]
            tmp_tourne1 = AX + BPrime + CX

            APrimeX = ['x' for _ in range(len(APrime))]
            CPrimeX = ['x' for _ in range(len(CPrime))]
            tmp_tourne2 = APrimeX + B + CPrimeX

            AX = [i if i not in BPrime else matrice_correspendance_Parent_1[i] for i in A]
            CX = [i if i not in BPrime else matrice_correspendance_Parent_1[i] for i in C]

            APrimeX = [i if i not in B else matrice_correspendance_Parent_2[i] for i in APrime]
            CPrimeX = [i if i not in B else matrice_correspendance_Parent_2[i] for i in CPrime]

            tmp_tourne1 = AX + BPrime + CX
            tmp_tourne2 = APrimeX + B + CPrimeX

            # Check and repair offspring routes to avoid repeating digits
            tmp_tourne1 = check_and_repair_route(tmp_tourne1)
            tmp_tourne2 = check_and_repair_route(tmp_tourne2)

            enfant_1, enfant_2 = [0] + tmp_tourne1 + [0], [0] + tmp_tourne2 + [0]
            enfants.append(enfant_1)
            enfants.append(enfant_2)

    return enfants, parents


def check_and_repair_route(route):
    repeated = set()
    repaired_route = []

    for node in route:
        if node not in repeated:
            repaired_route.append(node)
            repeated.add(node)
        else:
            for i in range(1, len(route) + 1):
                if i not in repeated:
                    repaired_route.append(i)
                    repeated.add(i)
                    break

    return repaired_route
# =================================================================================================================
# =================================================================================================================

def swap(enfants=[]):
    import random
    print(
        '**********************************************************************************************************************')
    print(
        '--------------------------------------------       OPERATION DE MUTATION      ----------------------------------------')
    print(
        '**********************************************************************************************************************')
    children = enfants.copy()

    # Pour chaque tournée de la population, soit 5
    for tournee in children:

        # print(f'Tournée : {tournee}')

        # Générer un nombre aléatoire r2 entre 0 et 1
        r1 = random.random()
        print('===================================')
        print(f'valeur de r1 = {r1}')
        print('===================================')

        # Si r1 <= 0.5 Alors
        # rendre la valeur de comparaison à 0.01
        if r1 < 0.01:
            children.remove(tournee)
            # Choisir 2 clients a et b de la tournée
            i = random.randint(1, len(tournee) - 2)
            j = random.randint(1, len(tournee) - 2)

            # tant que les indexes sont les mêmes, répéter l'opération de génération de j
            while j == i:
                j = random.randint(1, len(tournee) - 2)
            print('les clients sélectionnés : \n')
            print(f'client {tournee[i]} : index {i}, client {tournee[j]} : index {j}')

            print(f'Tournée initiale : {tournee}')
            # INVERSER LES CLIENTS SELECTIONNES
            tournee[i], tournee[j] = tournee[j], tournee[i]
            children.append(tournee)
            print(f'Tournée finale : {tournee}')
            print("------------------------------------------------------------------------")
    return children


# =================================================================================================================

def calculate_fitness_solution(ll):
    return sum([fitness(i, distance_matrix) for i in ll])


# =================================================================================================================


def genetic_algorithm(population_size, nb_iteration, nb_selected_parents):
    # Création de la population initiale :
    population = generate_init_population(population_size=population_size)

    tmp_enfants = []

    tmp_parents = []

    for _ in range(nb_iteration):
        print(f'leeeeeeeeeeeen parents : {len(population)}')

        # Appliquer la sélection par tournoi binaire
        population = selection(tours_geants=population, nb_parents=nb_selected_parents)

        # Appliquer le croisement
        enfants, population = croisement(selected_routes_input=population)
        tmp_parents = population

        # Appliquer la mutation
        enfants = swap(enfants=enfants)

        # obtenir les sous-tournees a partir des enfants (Fonction restante)
        for child in enfants:
            sub_ch = sub_children(child)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(child)
            print(sub_ch)
            print(creer_tour_geant(sub_ch))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            population.append((child, calculate_fitness_solution(sub_ch)))

    last_generation = population
    fitness_solution = []

    # calculer le fitness pour les derniers enfants
    for last_element in last_generation:
        subs = sub_children(last_element[0])
        f = sum([fitness(i, distance_matrix) for i in subs])
        fitness_solution.append((last_element[0], f, subs))

    # Choisir la tournée de plus petit fitness (coût)
    best_route = min(fitness_solution, key=lambda x: x[1])

    return best_route


# =================================================================================================================

import random

population_size = int(input('Taille de population : '))
nb_iteration = int(input('Nombre d\'itération de l\'algorithme génétique  : '))

nb_clients = int(input('Nombre de client  : '))

# nombre de parents a sélectionner dans l'operation de selection
nb_selected_parents = int(input('Nombre de parents à sélectionner dans la SELECTION : '))
# le nombre de parents à sélectionner doit etre inférieur ou égal la taille de population
while nb_selected_parents > population_size:
    nb_selected_parents = int(input('Nombre de parents à sélectionner dans la SELECTION : '))

max_dist = int(input('Max distance  : '))
nb_vehicules = int(input('Nombre de véhicules  : '))
max_capacity_vehicule = int(input('Max capacity vehicule: '))

# Appeler la  fonction d initialisation des données
(number_clients, commands,
 generated_distance_matrix,
 truck_capacity, max_distance, num_trucks) = init_data(number_clients=nb_clients,
                                                       truck_capacity=max_capacity_vehicule,
                                                       max_distance=max_dist,
                                                       num_trucks=nb_vehicules)

# A copier la liste des commandes
commands = [(1, 175), (2, 23), (3, 66), (4, 196), (5, 123), (6, 86), (7, 105), (8, 115), (9, 196), (10, 29), (11, 48),
            (12, 126), (13, 24), (14, 122), (15, 29), (16, 168), (17, 151), (18, 27), (19, 182), (20, 58), (21, 120),
            (22, 170), (23, 128), (24, 99), (25, 44), (26, 93), (27, 100), (28, 8), (29, 194)]

distance_matrix = read_dist_matrix_from_file(file='data.txt', delimiter=',')
# distance_matrix = generated_distance_matrix


best_solution = genetic_algorithm(population_size=population_size,
                                  nb_iteration=nb_iteration,
                                  nb_selected_parents=nb_selected_parents)

print(
    '===========================================================================--------------------------------------------===========================================================================')
print('best solution : ', best_solution)
print(
    '===========================================================================--------------------------------------------===========================================================================')

print('##################################################################################################')
print(f'SOLUTION : ')
print(f'SOUS TOURS : {best_solution[2]}')
print(f'NOMBRE DE SOUS TOURS: {len(best_solution[2])}')
print(f'FITNESS  : {best_solution[1]}')
print('##################################################################################################')
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================