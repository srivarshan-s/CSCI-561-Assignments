#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

// Global constant to declare the board size
const int BOARD_SIZE = 12;

// Global constants for board cells
const char WHITE = 'O';
const char BLACK = 'X';
const char EMPTY = '.';

// Global constants for alpha-beta pruning
const int ALPHA = -999;
const int BETA = 999;

// Global constants for dynamic depth
const int LOWER_DEPTH = 6;
const int UPPER_DEPTH = 10;
const int PIVOT = 10;

// Class to maintain game state
class GameState
{
private:
    vector<vector<char>> board;
    char player;
    char opponent;
    bool player_turn;

public:
    // Constructor
    GameState(vector<vector<char>> board, char player, char opponent, bool turn)
    {
        this->board = board;
        this->player = player;
        this->opponent = opponent;
        this->player_turn = turn;
    }

    // Function to return player_turn
    bool turn()
    {
        return this->player_turn;
    }

    // Function to print the state of the board
    void print_board()
    {
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                cout << this->board[i][j] << "    ";
            }
            cout << "\n\n";
        }
    }

    // Function to evaluate the state of the board
    int evaluate()
    {
        int val;
        // Count the number of white and black
        int num_white = 0;
        int num_black = 0;
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (this->board[i][j] == WHITE)
                    num_white++;
                if (this->board[i][j] == BLACK)
                    num_black++;
            }
        }
        // Since white always starts black get +1 bonus
        num_black++;
        // Value is difference between num of white and black
        if (player == WHITE)
            val = num_white - num_black;
        else
            val = num_black - num_white;
        return val;
    }

    // Function to return the valid moves
    vector<pair<int, int>> valid_moves()
    {
        // Initialize vector to store moves
        vector<pair<int, int>> moves;
        // Choose white or black to check acccording to player turn
        char cell_to_check, cell_to_flip;
        if (player_turn)
        {
            cell_to_check = this->opponent;
            cell_to_flip = this->player;
        }
        else
        {
            cell_to_check = this->player;
            cell_to_flip = this->opponent;
        }
        // Iterate through the board cells
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                // Check if cell is empty
                if (board[i][j] == EMPTY)
                {
                    bool flag = false;
                    int x, y;

                    // Check left of empty cell
                    if (j - 1 > 0 && board[i][j - 1] == cell_to_check)
                    {
                        // Check left cells till cell_to_flip is reached
                        for (y = j - 1; y >= 0; y--)
                        {
                            if (board[i][y] == EMPTY)
                                break;
                            if (board[i][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check right of empty cell
                    if (j + 1 < BOARD_SIZE - 1 && board[i][j + 1] == cell_to_check)
                    {
                        // Check right cells till cell_to_flip is reached
                        for (y = j + 1; y < BOARD_SIZE; y++)
                        {
                            if (board[i][y] == EMPTY)
                                break;
                            if (board[i][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check up of empty cell
                    if (i - 1 > 0 && board[i - 1][j] == cell_to_check)
                    {
                        // Check upper cells till cell_to_flip is reached
                        for (x = i - 1; x >= 0; x--)
                        {
                            if (board[x][j] == EMPTY)
                                break;
                            if (board[x][j] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check down of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && board[i + 1][j] == cell_to_check)
                    {
                        // Check lower cells till cell_to_flip is reached
                        for (x = i + 1; x < BOARD_SIZE; x++)
                        {
                            if (board[x][j] == EMPTY)
                                break;
                            if (board[x][j] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check top left of empty cell
                    if (i - 1 > 0 && j - 1 > 0 && board[i - 1][j - 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i - 1, y = j - 1; x >= 0 && y >= 0; x--, y--)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check top right of empty cell
                    if (i - 1 > 0 && j + 1 < BOARD_SIZE - 1 && board[i - 1][j + 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i - 1, y = j + 1; x >= 0 && y < BOARD_SIZE; x--, y++)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check bottom left of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && j - 1 > 0 && board[i + 1][j - 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i + 1, y = j - 1; x < BOARD_SIZE && y >= 0; x++, y--)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check bottom right of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && j + 1 < BOARD_SIZE - 1 && board[i + 1][j + 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i + 1, y = j + 1; x < BOARD_SIZE && y < BOARD_SIZE; x++, y++)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }
                }
            }
        }
        return moves;
    }

    // Function to apply a move
    GameState play(pair<int, int> move)
    {
        vector<vector<char>> new_board = this->board;

        char cell_to_flip = this->player_turn ? this->player : this->opponent;
        int x = move.first;
        int y = move.second;
        new_board[x][y] = cell_to_flip;

        // Check left of cell
        if (y - 1 > 0)
        {
            bool flag = false;
            int j;
            for (j = y - 2; j >= 0; j--)
            {
                if (new_board[x][j] == EMPTY)
                    break;
                if (new_board[x][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; j < y; j++)
                {
                    new_board[x][j] = cell_to_flip;
                }
            }
        }

        // Check right of cell
        if (y + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int j;
            for (j = y + 2; j < BOARD_SIZE; j++)
            {
                if (new_board[x][j] == EMPTY)
                    break;
                if (new_board[x][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; j > y; j--)
                {
                    new_board[x][j] = cell_to_flip;
                }
            }
        }

        // Check top of cell
        if (x - 1 > 0)
        {
            bool flag = false;
            int i;
            for (i = x - 2; i >= 0; i--)
            {
                if (new_board[i][y] == EMPTY)
                    break;
                if (new_board[i][y] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x; i++)
                {
                    new_board[i][y] = cell_to_flip;
                }
            }
        }

        // Check bottom of cell
        if (x + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int i;
            for (i = x + 2; i < BOARD_SIZE; i++)
            {
                if (new_board[i][y] == EMPTY)
                    break;
                if (new_board[i][y] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x; i--)
                {
                    new_board[i][y] = cell_to_flip;
                }
            }
        }

        // Check top left of cell
        if (x - 1 > 0 && y - 1 > 0)
        {
            bool flag = false;
            int i, j;
            for (i = x - 2, j = y - 2; i >= 0 && j >= 0; i--, j--)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x && j < y; i++, j++)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check top right of cell
        if (x - 1 > 0 && y + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int i, j;
            for (i = x - 2, j = y + 2; i >= 0 && j < BOARD_SIZE; i--, j++)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x && j > y; i++, j--)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check bottom left of cell
        if (x + 1 < BOARD_SIZE - 1 && y - 1 > 0)
        {
            bool flag = false;
            int i, j;
            for (i = x + 2, j = y - 2; i < BOARD_SIZE && j >= 0; i++, j--)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x && j < y; i--, j++)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check bottom right of cell
        if (x + 1 < BOARD_SIZE - 1 && y + 1 < BOARD_SIZE)
        {
            bool flag = false;
            int i, j;
            for (i = x + 2, j = y + 2; i < BOARD_SIZE && j < BOARD_SIZE; i++, j++)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x && j > y; i--, j--)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        GameState new_state(new_board, this->player, this->opponent, !this->player_turn);
        return new_state;
    }
};

// Function to perform minimax with alpha-beta pruning
pair<int, pair<int, int>> mini_max(GameState game_state, int depth, int limit, pair<int, int> prev_move, int alpha, int beta)
{
    // If depth reaches limit then return game state val
    if (depth >= limit)
    {
        return make_pair(game_state.evaluate(), prev_move);
    }
    
    // Get all valid moves
    vector<pair<int, int>> valid_moves = game_state.valid_moves();

    // Check if no more valid moves are possible
    if (valid_moves.empty())
    {
        return make_pair(game_state.evaluate(), prev_move);
    }

    // If the number of valid moves are greater than the pivot,
    // then reduce the search depth
    if (valid_moves.size() > PIVOT)
    {
        limit = LOWER_DEPTH;
        if (depth >= limit)
        {
            return make_pair(game_state.evaluate(), prev_move);
        }
    }

    // If player turn then find the max
    if (game_state.turn())
    {
        int max = -999;
        pair<int, int> next_move;
        for (pair<int, int> move : valid_moves)
        {
            int val = mini_max(game_state.play(move), depth + 1, limit, move, alpha, beta).first;
            if (val > max)
            {
                max = val;
                next_move = move;
            }
            alpha = std::max(alpha, max);
            if (alpha >= beta)
                break;
        }
        return make_pair(max, next_move);
    }

    // If opponent turn then find the min
    else
    {
        int min = 999;
        pair<int, int> next_move;
        for (pair<int, int> move : valid_moves)
        {
            int val = mini_max(game_state.play(move), depth + 1, limit, move, alpha, beta).first;
            if (val < min)
            {
                min = val;
                next_move = move;
            }
            beta = std::min(beta, min);
            if (alpha >= beta)
                break;
        }
        return make_pair(min, next_move);
    }
}

// Function to get the move using idx
string get_move(pair<int, int> move)
{
    // Initialize look-up vector
    vector<string> lookup_vec = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"};
    string x_coord = lookup_vec[move.second];
    string y_coord = to_string(move.first + 1);
    return (x_coord + y_coord);
}

// Function to play game against random agent
void play_against_random(GameState start_state)
{
    cout << "START STATE" << '\n';
    start_state.print_board();

    vector<pair<int, int>> moves = start_state.valid_moves();
    GameState prev_state = start_state;
    srand(time(0));
    while (!moves.empty())
    {
        pair<int, int> next_move = moves[0];
        cout << "NEXT MOVE" << '\n';
        if (prev_state.turn())
        {
            next_move = mini_max(start_state, 0, UPPER_DEPTH, make_pair(-1, -1), ALPHA, BETA).second;
        }
        else
        {
            next_move = moves[rand() % moves.size()];
        }
        GameState next_state = prev_state.play(next_move);
        next_state.print_board();
        prev_state = next_state;
        moves = prev_state.valid_moves();
    }
}

int main()
{
    // Read the input file
    ifstream input_file("input.txt");

    // Read the player and opponent variables
    string player, opponent;
    getline(input_file, player);
    opponent = player == "X" ? "O" : "X";

    // Read the time remaining
    string time_remain;
    double player_time, opponent_time;
    getline(input_file, time_remain);
    istringstream iss(time_remain);
    iss >> player_time >> opponent_time;

    // Read the state of the board
    vector<vector<char>> board;
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        string board_line_str;
        getline(input_file, board_line_str);
        vector<char> board_line;
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            board_line.push_back(board_line_str[j]);
        }
        board.push_back(board_line);
    }

    // Close the file
    input_file.close();

    // Initialize GameState object
    GameState start_state(board, player[0], opponent[0], true);

    // Search for best move
    pair<int, int> move;
    move = mini_max(start_state, 0, UPPER_DEPTH, make_pair(-1, -1), ALPHA, BETA).second;

    // Write move to output file
    ofstream output_file("output.txt");
    output_file << get_move(move) << "\n";
    output_file.close();

    // play_against_random(start_state);

    return 0;
}
