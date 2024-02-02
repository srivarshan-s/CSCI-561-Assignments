#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <queue>

using namespace std;

// Global function to store path length
int path_len = 0;

// Function to recursively print path to goal node
void print_path(int state, vector<int> &visited, vector<int> &visited_parent, vector<string> &node_name, ofstream &output_file, ofstream &pathlen_file)
{
    // End recursion if you reach last node
    if (state == -1)
    {
        return;
    }
    // Get parent index
    int idx = -1;
    for (int i = 0; i < visited.size(); i++)
    {
        if (visited[i] == state)
        {
            idx = i;
            break;
        }
    }
    // Recurse to parent
    print_path(visited_parent[idx], visited, visited_parent, node_name, output_file, pathlen_file);
    if (node_name[state] == "goal")
    {
        cout << node_name[state] << endl;
        output_file << node_name[state] << endl;
        cout << "Path Length: " << path_len << endl;
        pathlen_file << path_len << endl;
    }
    else
    {
        cout << node_name[state] << " ";
        output_file << node_name[state] << " ";
        path_len += 1;
    }
}

// Function to see if an element is in a queue
bool queue_contains(queue<int> q, int ele)
{
    while (!q.empty())
    {
        if (q.front() == ele)
            return true;
        q.pop();
    }
    return false;
}

// Function to see if an element is in the open queue
bool contains(queue<int> open, queue<int> parent, int ele, int ele_parent)
{
    while (!open.empty())
    {
        if (open.front() == ele && parent.front() == ele_parent)
            return true;
        open.pop();
        parent.pop();
    }
    return false;
}
// Overload contains function to see if an element is in the visited vector
bool contains(vector<int> visited, vector<int> visited_parent, int ele, int ele_parent)
{
    for (int i = 0; i < visited.size(); i++)
    {
        if (visited[i] == ele && visited_parent[i] == ele_parent) return true;
    }
    return false;
}

// Function to see if an element is in a vector
bool vector_contains(vector<int> v, int ele)
{
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i] == ele)
            return true;
    }
    return false;
}

// Function to perform breadth-first search
void bfs(vector< vector<int> > &adj_matrix, map<string, int> &node_num_map, int num_nodes, vector<string> &node_name, int energy_limit, vector<int> &z_pos)
{
    // Initialize open queue
    queue<int> open;
    // It's parent queue
    queue<int> parent;

    // Initialize the momentum queue
    queue<int> momentum;

    // Initialize visited array
    vector<int> visited;
    // Initialize it's parent array
    vector<int> visited_parent;

    // Add start node to open queue
    open.push(node_num_map["start"]);
    // Add NULL to parent queue
    parent.push(-1);
    // Add 0 to momentum queue
    momentum.push(0);

    // Get goal node number
    int goal_node_num = node_num_map["goal"];

    // Main loop
    while (true)
    {
        // If open queue is empty then return
        if (open.empty())
        {
            ofstream output_file, pathlen_file;
            output_file.open("output.txt");
            output_file << "FAIL" << endl;
            pathlen_file.open("pathlen.txt");
            pathlen_file << -1 << endl;
            cout << "FAIL" << endl;
            output_file.close();
            pathlen_file.close();
            break;
        }

        // Get the front node
        int curr_node = open.front();
        open.pop();
        // Get parent of front node
        int curr_node_parent = parent.front();
        parent.pop();
        // Get momentum of front node
        int curr_momentum = momentum.front();
        momentum.pop();
        // Add to visited and visited_parent arrays
        visited.push_back(curr_node);
        visited_parent.push_back(curr_node_parent);

        // If we have reached the goal state then return
        if (curr_node == goal_node_num)
        {
            cout << "Goal reached! " << endl;
            ofstream output_file, pathlen_file;
            output_file.open("output.txt");
            pathlen_file.open("pathlen.txt");
            print_path(curr_node, visited, visited_parent, node_name, output_file, pathlen_file);
            output_file.close();
            pathlen_file.close();
            break;
        }

        // Expand the current node
        for (int i = 0; i < num_nodes; i++)
        {
            if (adj_matrix[curr_node][i] == 1)
            {                   
                // Check if child is already visited
                // if (vector_contains(visited, i))
                if (contains(visited, visited_parent, i, curr_node))
                    continue;

                // Check if child is already in open queue
                if (contains(open, parent, i, curr_node))
                    continue;

                // Check if node is reachable
                int energy_req = z_pos[i] - z_pos[curr_node];
                if (curr_momentum + energy_limit < energy_req)
                    continue;

                if (node_name[curr_node] == "uekv" && node_name[i] == "goal") 
                    cout << "present" << endl;

                // Add child to open queue
                open.push(i);
                parent.push(curr_node);

                // Calculate momentum for the child
                int child_momentum = max(0, z_pos[curr_node] - z_pos[i]);
                // Add momemtum to queue
                momentum.push(child_momentum);
            }
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

    // Read the uphill enery limit
    string rover_energy_str;
    getline(input_file, rover_energy_str);
    int rover_energy = stoi(rover_energy_str);

    // Read the number of nodes
    string num_nodes_str;
    getline(input_file, num_nodes_str);
    int num_nodes = stoi(num_nodes_str);

    // Iterate through all the nodes
    vector<string> node_name;
    vector<int> x_pos, y_pos, z_pos;
    map<string, int> node_num_map;
    for (int i = 0; i < num_nodes; i++)
    {
        string node_str;
        getline(input_file, node_str);
        istringstream iss(node_str);
        string name;
        int x, y, z;
        iss >> name >> x >> y >> z;
        node_name.push_back(name);
        x_pos.push_back(x);
        y_pos.push_back(y);
        z_pos.push_back(z);
        node_num_map[name] = i;
    }

    // Initialize an adjaceny matrix
    vector< vector<int> > adj_matrix;
    for (int i = 0; i < num_nodes; i++)
    {
        vector<int> temp;
        for (int j = 0; j < num_nodes; j++)
        {
            temp.push_back(0);
        }
        adj_matrix.push_back(temp);
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
        // Assign 1 to the connected nodes
        adj_matrix[node_num_map[node1]][node_num_map[node2]] = 1;
        adj_matrix[node_num_map[node2]][node_num_map[node1]] = 1;
    }

    // Close the input file
    input_file.close();

    // Perform search algorithm according to search type
    if (search_type == "BFS")
    {
        bfs(adj_matrix, node_num_map, num_nodes, node_name, rover_energy, z_pos);
    }

    return 0;
}