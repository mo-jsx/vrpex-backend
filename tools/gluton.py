class Truck:
    def __init__(self, capacity, max_distance):
        self.capacity = capacity
        self.commands = []
        self.weight = 0
        self.max_distance = max_distance
        self.current_distance = 0

    def can_add_command(self, command_id, command_weight, distance_matrix):
        return self.weight + command_weight <= self.capacity and (
            not self.commands
            or self.current_distance + distance_matrix[self.commands[-1]][command_id]
            <= self.max_distance
        )

    def add_command(self, command_id, command_weight, distance_matrix):
        if self.commands:
            self.current_distance += distance_matrix[self.commands[-1]][command_id]
        self.commands.append(command_id)
        self.weight += command_weight

    def exceeds_max_distance(self, distance_matrix):
        if self.commands:
            return (
                self.current_distance + distance_matrix[self.commands[-1]][0]
                > self.max_distance
            )
        return False


class CommandAllocator:
    def __init__(
        self, commands, truck_capacity, distance_matrix, max_distance, num_trucks
    ):
        self.commands = sorted(commands, key=lambda x: x[1], reverse=True)
        self.truck_capacity = truck_capacity
        self.distance_matrix = distance_matrix
        self.max_distance = max_distance
        self.num_trucks = num_trucks
        self.trucks = []
        self.unassigned_commands = []
        self.depot = 0

    def assign_trucks(self):
        for command_id, command_weight in self.commands:
            assigned = False
            for truck in self.trucks:
                if truck.can_add_command(
                    command_id, command_weight, self.distance_matrix
                ) and not truck.exceeds_max_distance(self.distance_matrix):
                    truck.add_command(command_id, command_weight, self.distance_matrix)
                    assigned = True
                    break

            if not assigned:
                if len(self.trucks) < self.num_trucks:
                    new_truck = Truck(self.truck_capacity, self.max_distance)
                    new_truck.add_command(
                        command_id, command_weight, self.distance_matrix
                    )
                    self.trucks.append(new_truck)
                else:
                    self.unassigned_commands.append((command_id, command_weight))

    def print_truck_assignments(self):
        tours = []
        for i, truck in enumerate(self.trucks):
            tour = []
            if truck.commands:
                tour.append(self.depot)
                for command_id in truck.commands:
                    tour.append(command_id+1)
                tour.append(self.depot)
                tours.append(tour)
        return tours

    def print_unassigned_commands(self):
        if self.unassigned_commands:
            print("Unassigned commands due to insufficient trucks:")
            for command_id, _ in self.unassigned_commands:
                print(f"Command {command_id}")

    def assign_trucks_and_return_results(self):
        self.assign_trucks()
        tours = self.print_truck_assignments()
        self.print_unassigned_commands()
        return tours


# Example usage
def gluton_algorithm(commands, truck_capacity, distance_matrix, max_distance, num_trucks):
  allocator = CommandAllocator(
      commands, truck_capacity, distance_matrix, max_distance, num_trucks
  )
  tours = allocator.assign_trucks_and_return_results()
  return tours
