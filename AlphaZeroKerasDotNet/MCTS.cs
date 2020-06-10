using ConnectFour;
using Numpy;
using System;
using System.Collections.Generic;

namespace AlphaZeroKerasDotNet
{
    public class Node
    {
        public GameState state;
        public int playerTurn;
        public string Id;
        public Dictionary<int, Edge> edges;

        public Node(GameState state)
        {
            this.state = state;
            this.playerTurn = state.PlayerTurn;
            this.Id = state.Id;
            this.edges = new Dictionary<int, Edge>();
        }

        public Boolean isLeaf()
        {
            return (edges.Count == 0);
        }
    }

    public class Edge
    {
        public int playerTurn;
        public string Id;
        public Node inNode;
        public Node outNode;
        public int action;
        public Dictionary<String, float> stats;

        public Edge(Node inNode, Node outNode, float prior, int action)
        {
            this.Id = inNode.state.Id + '|' + outNode.state.Id;
            this.inNode = inNode;
            this.outNode = outNode;
            this.playerTurn = inNode.state.PlayerTurn;
            this.action = action;
            stats = new Dictionary<String, float>() { { "N", 0 }, { "W", 0 }, { "Q", 0 }, { "P", prior } };
        }

    }

    public class MCTS
    {
        public Node root;
        public Dictionary<String, Node> tree;
        public int cpuct;

        public MCTS(Node root, int cpuct)
        {
            this.root = root;
            this.cpuct = cpuct;
            this.tree = new Dictionary<String, Node>();
            this.addNode(root);
        }

        public Tuple<Node, float, int, List<Edge>> moveToLeaf()
        {
            Node currentNode = this.root;
            List<Edge> breadcrumbs = new List<Edge>();
            int done = 0;
            float value = 0;
            float maxQU, Nb;
            float epsilon; 
            NDarray nu = null;
            while (!currentNode.isLeaf()) 
            {
                maxQU = -99999;
                if (currentNode == this.root)
                {
                    epsilon = Config.EPSILON;
                    List<float> nuList = new List<float>();
                    for (int i = 0; i <= currentNode.edges.Count - 1; i++)
                        nuList.Add(Config.ALPHA);
                    nu = np.array(nuList.ToArray());
                    nu = np.random.dirichlet(nu);
                }
                else
                {
                    epsilon = 0;
                    List<float> nuList = new List<float>();
                    for (int i = 0; i <= currentNode.edges.Count - 1; i++)
                        nuList.Add(0);
                    nu = np.array(nuList.ToArray());
                }
                Nb = 0;
                foreach (KeyValuePair<int, Edge> kvp in currentNode.edges)
                {
                    Nb += kvp.Value.stats["N"];
                }

                int idx = 0;
                int simulationAction = -1;
                Edge simulationEdge = null;
                foreach (KeyValuePair<int, Edge> kvp in currentNode.edges)
                {
                    int action = kvp.Key;
                    Edge edge = kvp.Value;
                    double U = this.cpuct * ((1-epsilon) * edge.stats["P"] + epsilon * (float)nu[idx]) * Math.Sqrt(Nb) / (1 + edge.stats["N"]);
                    float Q = edge.stats["Q"];
                    if (Q + U > maxQU)
                    {
                        maxQU = Q + (float)U;
                        simulationAction = action;
                        simulationEdge = edge;
                    }
                    idx += 1;
                }
                GameState newState = currentNode.state.TakeAction(simulationAction);
                currentNode = simulationEdge.outNode;
                value = newState.Value;
                done = Convert.ToInt32(newState.Done);

                breadcrumbs.Add(simulationEdge);
            };
            return new Tuple<Node, float, int, List<Edge>>(currentNode, value, done, breadcrumbs);
        }

        public void backFill(Node leaf, float value, List<Edge> breadcrumbs)
        {
            int currentPlayer = leaf.state.PlayerTurn;

            foreach (Edge edge in breadcrumbs)
            {
                int playerTurn = edge.playerTurn;
                int direction = (playerTurn == currentPlayer) ? 1 : -1;
                edge.stats["N"] = edge.stats["N"] + 1;
                edge.stats["W"] = edge.stats["W"] + value * direction;
                edge.stats["Q"] = edge.stats["W"] / edge.stats["N"];
            }
        }

        public void addNode(Node node)
        {
            this.tree[node.Id] = node;
        }
    }
}
