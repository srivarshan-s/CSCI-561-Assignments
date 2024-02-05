#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <queue>
#include <deque>
#include <vector>
#include<iomanip>
#include<math.h>

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
    float path_len;
    float estm_total_path_len;
};

// Custom comparator for UCS priority queue
struct UcsNodeComparator
{
    bool operator()(const Node &a, const Node &b)
    {
        return (a.path_len > b.path_len);
    }
};

// Custom comparator for A* priority queue
struct AStarNodeComparator
{
    bool operator()(const Node &a, const Node &b)
    {
        return (a.estm_total_path_len > b.estm_total_path_len);
    }
};

// Custom priority queue for UCS
class custom_priority_queue: public priority_queue<Node, vector<Node>, UcsNodeComparator> {
public:
    vector<Node> &impl() { return c; }
};

// Struct to store output file info
struct Output
{
    ofstream output_file;
    ofstream pathlen_file;
};

// Overloaded function to check if vector or queue have an element
bool contains(vector<Node> &vec, map< int, vector<int> > &vec_map, int &ele, int &parent, int &momemtum)
{
    for (int i = 0; i < vec_map[ele].size(); i++)
    {
        if (vec[vec_map[ele][i]].id == ele)
        {
            if (vec[vec_map[ele][i]].parent == parent)
                return true;
            if (vec[vec_map[ele][i]].momentum >= momemtum)
                return true;
        }
    }
    return false;
}
bool contains(map< int, vector<Node> > &m, int &ele, int &parent, int &momemtum)
{
    for (int i = 0; i < m[ele].size(); i++)
    {
        if (m[ele][i].id == ele)
        {
            if (m[ele][i].parent == parent)
                return true;
            if (m[ele][i].momentum >= momemtum)
                return true;
        }
    }
    return false;
}
bool contains_ucs(map< int, vector<Node> > &cpq_map, int &ele, int &parent, int &momemtum, float &path_len)
{
    for (int i = 0; i < cpq_map[ele].size(); i++)
    {
       if (cpq_map[ele][i].path_len <= path_len)
        {
            if (cpq_map[ele][i].momentum >= momemtum) return true;
        }
    }
    return false;
}
bool contains_ucs(vector<Node> &vec, map< int, vector<int> > &vec_map, int &ele, int &parent, int &momemtum, float &path_len)
{
    for (int i = 0; i < vec_map[ele].size(); i++)
    {
       if (vec[vec_map[ele][i]].path_len <= path_len)
        {
            if (vec[vec_map[ele][i]].momentum >= momemtum) return true;
        }
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
    deque<Node> open;
    // Initialize map for open queue
    map< int, vector<Node> > open_map;

    // Initialize visited array
    vector<Node> visited;
    // Initialize map for visited array
    map< int, vector<int> > visited_map;

    // Add start node to open queue
    Node start_node = node_list[node_num_map["start"]];
    start_node.parent = -1;
    start_node.path_len = 0;
    start_node.momentum = 0;
    open.push_back(start_node);
    open_map[start_node.id].push_back(start_node);

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
        open.pop_front();

        // Add node to visited array
        visited.push_back(curr_node);
        visited_map[curr_node.id].push_back(visited.size() - 1);

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
            bool x = contains(visited, visited_map, child, parent_idx, child_node.momentum);
            end = clock();
            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            close_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is in open queue
            start = clock();
            x = contains(open_map, child, parent_idx, child_node.momentum);
            end = clock();
            time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            open_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is reachable
            int energy_req = child_node.z - curr_node.z;
            if (curr_node.momentum + energy_limit < energy_req)
                continue;

            // Compute parent of the child
            child_node.parent = parent_idx;

            // Compute path length for the child
            child_node.path_len = curr_node.path_len + 1;

            // Add child to open queue
            open.push_back(child_node);
            open_map[child_node.id].push_back(child_node);
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

// Function to compute euclidean distance
double euc_dist(int x1, int y1, int x2, int y2)
{
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}
double euc_dist(int x1, int y1, int z1, int x2, int y2, int z2)
{
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2) + pow((z2 - z1), 2));
}

// Function to perform uniform-cost search
void ucs(vector<Node> &node_list, map<string, int> &node_num_map, int energy_limit, Output &output)
{
    // Initialize clock functions
    clock_t start, end;
    vector<double> open_time, close_time;

    // Initialize open priority queue
    custom_priority_queue open;
    // Initialize map for open queue
    map< int, vector<Node> > open_map;

    // Initialize visited array
    vector<Node> visited;
    // Initialize map for visited array
    map< int, vector<int> > visited_map;

    // Add start node to open queue
    Node start_node = node_list[node_num_map["start"]];
    start_node.parent = -1;
    start_node.path_len = 0;
    start_node.momentum = 0;
    open.push(start_node);
    open_map[start_node.id].push_back(start_node);

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

        // Get the front node from open priority queue
        Node curr_node = open.top();
        open.pop();

        // Add node to visited array
        visited.push_back(curr_node);
        // Add to visited map
        visited_map[curr_node.id].push_back(visited.size() - 1);

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

            // Compute path length for the child
            child_node.path_len = curr_node.path_len + euc_dist(curr_node.x, curr_node.y, child_node.x, child_node.y);

            // Check if child is in visited array
            start = clock();
            bool x = contains_ucs(visited, visited_map, child, parent_idx, child_node.momentum, child_node.path_len);
            end = clock();
            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            close_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is in open priority queue
            start = clock();
            x = contains_ucs(open_map, child, parent_idx, child_node.momentum, child_node.path_len);
            end = clock();
            time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            open_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is reachable
            int energy_req = child_node.z - curr_node.z;
            if (curr_node.momentum + energy_limit < energy_req)
                continue;

            // Compute parent of the child
            child_node.parent = parent_idx;

            // Add child to open queue
            open.push(child_node);
            open_map[child_node.id].push_back(child_node);
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

// Evaluation function for heuristic search
float eval_func(Node &n1, Node &n2)
{
    return euc_dist(n1.x , n1.y, n1.z, n2.x , n2.y, n2.z);
}

// Function to perform a* search
void a_star(vector<Node> &node_list, map<string, int> &node_num_map, int energy_limit, Output &output)
{
    // Initialize clock functions
    clock_t start, end;
    vector<double> open_time, close_time;

    // Initialize open priority queue
    priority_queue<Node, vector<Node>, AStarNodeComparator> open;
    // Initialize map for open queue
    map< int, vector<Node> > open_map;

    // Initialize visited array
    vector<Node> visited;
    // Initialize map for visited array
    map< int, vector<int> > visited_map;

    // Get the goal node for the heuristic function
    Node goal_node = node_list[node_num_map["goal"]];

    // Add start node to open queue
    Node start_node = node_list[node_num_map["start"]];
    start_node.parent = -1;
    start_node.path_len = 0;
    start_node.estm_total_path_len = eval_func(start_node, goal_node);
    start_node.momentum = 0;
    open.push(start_node);
    open_map[start_node.id].push_back(start_node);

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

        // Get the front node from open priority queue
        Node curr_node = open.top();
        open.pop();

        // Add node to visited array
        visited.push_back(curr_node);
        // Add to visited map
        visited_map[curr_node.id].push_back(visited.size() - 1);

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

            // Compute path length for the child
            child_node.path_len = curr_node.path_len + euc_dist(curr_node.x, curr_node.y, curr_node.z, child_node.x, child_node.y, child_node.z);

            // Check if child is in visited array
            start = clock();
            bool x = contains_ucs(visited, visited_map, child, parent_idx, child_node.momentum, child_node.path_len);
            end = clock();
            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            close_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is in open priority queue
            start = clock();
            x = contains_ucs(open_map, child, parent_idx, child_node.momentum, child_node.path_len);
            end = clock();
            time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            open_time.push_back(time_taken);
            if (x)
                continue;

            // Check if child is reachable
            int energy_req = child_node.z - curr_node.z;
            if (curr_node.momentum + energy_limit < energy_req)
                continue;

            // Compute parent of the child
            child_node.parent = parent_idx;

            // Compute estimated total path length
            child_node.estm_total_path_len = child_node.path_len + eval_func(child_node, goal_node);

            // Add child to open queue
            open.push(child_node);
            open_map[child_node.id].push_back(child_node);
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
    Output output;
    output.output_file.open("output.txt");
    output.pathlen_file.open("pathlen.txt");

    // Perform search algorithm according to search type
    if (search_type == "BFS")
    {
        bfs(node_list, node_num_map, rover_energy, output);
    }
    else if (search_type == "UCS")
    {
        ucs(node_list, node_num_map, rover_energy, output);
    }
    else if (search_type == "A*")
    {
        a_star(node_list, node_num_map, rover_energy, output);

    }

    // Close the files in output struct
    output.output_file.close();
    output.pathlen_file.close();

    return 0;
}