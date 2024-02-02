#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <queue>

using namespace std;

struct Node
{
    string name;
    int x, y, z;
    vector<string> neighbors;
    string parent;
    int momentum;
    int path_len;
};

void print_soln(string state, map<string, Node>& state_map, ofstream& op_file, ofstream& pl_file)
{
    if (state == "NULL")
    {
        return;
    }
    print_soln(state_map[state].parent, state_map, op_file, pl_file);
    if (state == "goal")
    {
        cout << state << endl;
        op_file << state << endl;
        cout << "Path Length: " << state_map[state].path_len << endl;
        pl_file << state_map[state].path_len << endl;
    } else {
        cout << state << " ";
        op_file << state << " ";
    }
}

void bfs(map<string, Node>& node_map, int energy_limit)
{
    // Initialize the open and close queues
    queue<Node> open;
    map<string, int> open_map;
    map<string, Node> close_map;

    open.push(node_map["start"]);
    open_map["start"] = 0;

    // Main loop
    while (true)
    {
        // If open is empty then return
        if (open.empty())
        {
            ofstream output_file;
            output_file.open("output.txt");
            output_file << "FAIL" << endl;
            cout << "FAIL" << endl;
            break;
        }
        
        // Get the front node
        Node curr_node = open.front();
        open.pop();
        close_map[curr_node.name] = curr_node;
        
        // If we have reached goal state then return
        if (curr_node.name == "goal")
        {
            string path = curr_node.name;
            ofstream output_file, pathlen_file;
            output_file.open("output.txt");
            pathlen_file.open("pathlen.txt");
            print_soln(path, close_map, output_file, pathlen_file);
            output_file.close();
            pathlen_file.close();
            break;
        }
        
        // Expand the current node
        vector<string> children = curr_node.neighbors;
        while (!children.empty())
        {
            Node child = node_map[children.back()];
            children.pop_back();

            // Check if child exists in close map
            bool is_in = close_map.find(child.name) != close_map.end();
            if (is_in)
            {
                // Skip child if it already exists in the map
                continue;
            }

            // Check if child exists in open queue
            // if (queue_contains(open, child.name))
            is_in = open_map.find(child.name) != open_map.end();
            if (is_in)
            {
                // Skip child if it already exists in the map
                continue;
            }

            // Check if move is valid
            int energy_req = child.z - curr_node.z;
            if (curr_node.momentum + energy_limit < energy_req)
            {
                // Skip child if move is invalid
                continue;
            }
            // Calculate the momentum for the child
            child.momentum = max(0, curr_node.z - child.z);
            
            // Add the child to open queue
            child.parent = curr_node.name;
            child.path_len = curr_node.path_len + 1;
            open.push(child);
            open_map[child.name] = 0;
        }
        
    }
    
    

}

int main()
{
    // Read the input file
    ifstream input_file("input.txt");

    // Read the type of search
    string search_type;
    getline(input_file, search_type);

    // Read the Rover's Uphill Energy Limit
    string rover_energy_str;
    getline(input_file, rover_energy_str);
    int rover_energy = stoi(rover_energy_str);

    // Read the number of nodes
    string num_nodes_str;
    getline(input_file, num_nodes_str);
    int num_nodes = stoi(num_nodes_str);

    // Iterate through all nodes
    map<string, Node> node_map;
    for (int i = 0; i < num_nodes; i++)
    {
        string node_str;
        getline(input_file, node_str);
        istringstream iss(node_str);
        string name;
        int x, y, z;
        Node node;
        iss >> node.name >> node.x >> node.y >> node.z;
        node.parent = "NULL";
        node.momentum = 0;
        node.path_len = 0;
        node_map[node.name] = node;
    }

    // Read the number of safe paths
    string num_safe_path_str;
    getline(input_file, num_safe_path_str);
    int num_safe_path = stoi(num_safe_path_str);

    // Iterate through all safe paths
    for (int i = 0; i < num_safe_path; i++)
    {
        string safe_path_str;
        getline(input_file, safe_path_str);
        istringstream iss(safe_path_str);
        string node1, node2;
        iss >> node1 >> node2;
        // Add the nodes in the safe path as neoghbors
        node_map[node1].neighbors.push_back(node2);
        node_map[node2].neighbors.push_back(node1);
    }

    input_file.close();

    // Perform search algorithm according to search type
    if (search_type == "BFS")
    {
        bfs(node_map, rover_energy);
    }

    return 0;
}
