def main():
    # Read the input file
    input_file = open("input.txt", "r")
    
    # Read the type of search
    search_type: str = input_file.readline().replace("\n", "")

    # Read the rover's uphill energy limit
    rover_energy: int = int(input_file.readline())

    # Read the number of nodes
    num_nodes: int = int(input_file.readline())

    # Iterate through all nodes
    node_names: list[str] = []
    x_pos: list[int] = []
    y_pos: list[int] = []
    z_pos: list[int] = []
    node_num_map: dict[str, int] = {}
    for idx in range(num_nodes):
        node_details: list[str] = input_file.readline().replace("\n", "").split(" ")
        node_names.append(node_details[0])
        node_num_map[node_details[0]] = idx
        x_pos.append(node_details[1])
        y_pos.append(node_details[2])
        z_pos.append(node_details[3])

    # Initialize an adjacency matrix
    adj_matrix: list[list[int]] = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Read the number of edges
    num_edges: int = int(input_file.readline())
    for _ in range(num_edges):
        node_details: list[str] = input_file.readline().replace("\n", "").split(" ")
        node1 = node_num_map[node_details[0]]
        node2 = node_num_map[node_details[1]]
        print(node1, node2)
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1


if __name__ == "__main__":
    main()