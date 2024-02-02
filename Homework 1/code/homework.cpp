#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <queue>

using namespace std;

// Struct to store node data
struct Node
{
    string name;
    int id;
    int x, y, z;
    vector<int> neighbors;
    int parent;
    int momentum;
    int path_len;
};

// Struct to store output file info
struct Output
{
    ofstream output_file;
    ofstream pathlen_file;
};

// Overloaded function to check if vector or queue have an element
bool contains(vector<Node> &vec, int &ele, int &parent, int &momemtum)
{
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i].id == ele)
        {
            if (vec[i].parent == parent)
                return true;
            if (vec[i].momentum >= momemtum)
                return true;
        }
    }
    return false;
}
bool contains(queue<Node> q, int &ele, int &parent, int &momemtum)
{
    while (!q.empty())
    {
        if (q.front().id == ele)
        {
            if (q.front().parent == parent)
                return true;
            if (q.front().momentum >= momemtum)
                return true;
        }
        q.pop();
    }
    return false;
}

// Function to print the path
void print_path(vector<Node> &visited, Node &node, Output &output)
{
    // End recursion of you reach the last node
    if (node.parent == -1)
    {
        cout << node.name << " ";
        output.output_file << node.name << " ";
        return;
    }

    // Recurse to the parent
    Node parent = visited[node.parent];
    print_path(visited, parent, output);

    // Print node name
    if (node.name == "goal")
    {
        cout << node.name << endl;
        output.output_file << node.name << endl;
        cout << "Path length: " << node.path_len << endl;
        output.pathlen_file << node.path_len << endl;
    }
    else
    {
        cout << node.name << " ";
        output.output_file << node.name << " ";
    }
}

// Function to perform breadth-first search
void bfs(vector<Node> &node_list, map<string, int> &node_num_map, int energy_limit, Output &output)
{
    // Initialize clock functions
    clock_t start, end;
    vector<double> open_time, close_time;
    
    // Initialize open queue
    queue<Node> open;

    // Initialize visited array
    vector<Node> visited;

    // Add start node to open queue
    Node start_node = node_list[node_num_map["start"]];
    start_node.parent = -1;
    start_node.path_len = 0;
    start_node.momentum = 0;
    open.push(start_node);

    // Main loop
    while (true)
    {
        // If open queue is empty then return FAIL
        if (open.empty())
        {
            output.output_file << "FAIL" << endl;
            cout << "FAIL" << endl;
            output.pathlen_file << -1 << endl;
            cout << "Path lenght: " << -1 << endl;
            break;
        }

        // Get the front node from open queue
        Node curr_node = open.front();
        open.pop();

        // Add node to visited array
        visited.push_back(curr_node);

        // If we have reached goal state then return path
        if (curr_node.name == "goal")
        {
            cout << "Goal reached!" << endl;
            print_path(visited, curr_node, output);
            break;
        }

        // Expand the current node
        vector<int> children = curr_node.neighbors;
        for (int i = 0; i < children.size(); i++)
        {
            int child = children[i];
            int parent_idx = visited.size() - 1;

            // Get the child node
            Node child_node = node_list[child];

            // Calculate momentum for the child
            child_node.momentum = max(0, curr_node.z - child_node.z);

            // Check if child is in visited array
            start = clock();
            bool x = contains(visited, child, parent_idx, child_node.momentum);
            end = clock();
            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            close_time.push_back(time_taken);
            if (x) continue;

            // Check if child is in open queue
            start = clock();
            x = contains(open, child, parent_idx, child_node.momentum);
            end = clock();
            time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            open_time.push_back(time_taken);
            if (x) continue;

            // Check if child is reachable
            int energy_req = child_node.z - curr_node.z;
            if (curr_node.momentum + energy_limit < energy_req)
                continue;

            // Compute parent of the child
            child_node.parent = parent_idx;

            // Compute path length for the child
            child_node.path_len = curr_node.path_len + 1;

            // Add child to open queue
            open.push(child_node);
        }
    }
    
    double time_taken_open = 0;
    for (int i = 0; i < open_time.size(); i++)
    {
        time_taken_open += open_time[i];
    }

    double time_taken_close = 0;
    for (int i = 0; i < close_time.size(); i++)
    {
        time_taken_close += close_time[i];
    }

    cout << endl;
    cout << "Total time taken for open queue is : " << fixed 
         << time_taken_open << setprecision(5);
    cout << " sec " << endl;
    cout << "Total time taken for close array is : " << fixed 
         << time_taken_close << setprecision(5);
    cout << " sec " << endl;
}

int main()
{
    // Read the input file
    ifstream input_file("input.txt");

    // Read the type of search
    string search_type;
    getline(input_file, search_type);

    // Read the uphill enery limit
    string rover_energy_str;
    getline(input_file, rover_energy_str);
    int rover_energy = stoi(rover_energy_str);

    // Read the number of nodes
    string num_nodes_str;
    getline(input_file, num_nodes_str);
    int num_nodes = stoi(num_nodes_str);

    // Create a node name to id mapping
    map<string, int> node_num_map;

    // Iterate through all the nodes
    vector<Node> node_list;
    for (int id = 0; id < num_nodes; id++)
    {
        string node_str;
        getline(input_file, node_str);
        istringstream iss(node_str);
        Node node;
        iss >> node.name >> node.x >> node.y >> node.z;
        node.id = id;
        node_num_map[node.name] = id;
        // Add node to node list
        node_list.push_back(node);
    }

    // Read the number of edges
    string num_edges_str;
    getline(input_file, num_edges_str);
    int num_edges = stoi(num_edges_str);

    // Read the edges
    for (int i = 0; i < num_edges; i++)
    {
        string edge_str;
        getline(input_file, edge_str);
        istringstream iss(edge_str);
        string node1, node2;
        iss >> node1 >> node2;
        // Find the node ids
        int node1_id = node_num_map[node1];
        int node2_id = node_num_map[node2];
        // Assign both nodes as their corresponding neighbors
        node_list[node1_id].neighbors.push_back(node2_id);
        node_list[node2_id].neighbors.push_back(node1_id);
    }

    // Close the input file
    input_file.close();

    // // Initialize the output struct
    // Output output;
    // output.output_file.open("output.txt");
    // output.pathlen_file.open("pathlen.txt");

    // Perform search algorithm according to search type
    if (search_type == "BFS")
    {
        // Initialize the output struct
        Output output;
        output.output_file.open("output.txt");
        output.pathlen_file.open("pathlen.txt");

        bfs(node_list, node_num_map, rover_energy, output);

        output.output_file.close();
        output.pathlen_file.close();
    }

    // Close the files in output struct
    // output.output_file.close();
    // output.pathlen_file.close();

    return 0;
}