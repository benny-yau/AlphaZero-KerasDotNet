using ConnectFour;
using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AlphaZeroKerasDotNet
{
    public class User
    {
        public String name;
        public int state_size;
        public int action_size;
        public MCTS mcts;
        public User(String name, int state_size, int action_size)
        {
            this.name = name;
            this.state_size = state_size;
            this.action_size = action_size;
        }

        public virtual int Act(GameState state, int tau)
        {
            int number;
            while (true)
            {
                Console.WriteLine("Enter your chosen action (type \'help\' to find index): ");
                String action = Console.ReadLine();

                if (action == "help")
                    GameState.DisplayHelp();
                else
                {
                    if (Int32.TryParse(action, out number))
                    {
                        if (state.AllowedActions.Contains(number))
                            break;
                    }
                }

            }
            return number;
        }

    }
    public class Agent : User
    {
        public int cpuct;
        public int MCTSsimulations;
        public Residual_CNN model;
        public Node root;

        public Agent(String name, int state_size, int action_size, int mcts_simulations, int cpuct, Residual_CNN model)
            : base(name, state_size, action_size)
        {
            this.name = name;
            this.state_size = state_size;
            this.action_size = action_size;
            this.cpuct = cpuct;
            this.MCTSsimulations = mcts_simulations;
            this.model = model;
            this.mcts = null;
        }

        public override int Act(GameState state, int tau)
        {
            if (this.mcts == null || !this.mcts.tree.ContainsKey(state.Id))
                this.buildMCTS(state);
            else
                this.changeRootMCTS(state);

            for (int sim = 0; sim <= this.MCTSsimulations - 1; sim++)
            {
                this.simulate();
            }
            Tuple<NDarray, NDarray> tuple = this.getAV(1);
            NDarray pi = tuple.Item1;
            NDarray values = tuple.Item2;
            int action = this.chooseAction(pi, values, tau);
            //state.TakeAction(action);
            return action;
        }

        public void simulate()
        {
            Tuple<Node, float, int, List<Edge>> tuple = this.mcts.moveToLeaf();
            Node leaf = tuple.Item1;
            float value = tuple.Item2;
            int done = tuple.Item3;
            List<Edge> breadcrumbs = tuple.Item4;
            value = this.evaluateLeaf(leaf, value, done);
            this.mcts.backFill(leaf, value, breadcrumbs);
        }

        public Tuple<dynamic, NDarray> get_preds(GameState state)
        {
            NDarray inputToModel = np.array(new[] { this.model.convertToModelInput(state) });
            dynamic preds = this.model.model.Predict(inputToModel).PyObject;
            var value_array = preds[0];
            var logits_array = preds[1];
            var value = value_array[0];
            var logits = logits_array[0]; //Python.Runtime.PyObject

            NDarray logitsArray = new NDarray(logits); //logitsArray.GetData<double>()
            
            for (int i = 0; i <= logitsArray.len - 1; i++)
            {
                if (!state.AllowedActions.Contains(i))
                    logitsArray[i] = np.array(-100);
            }

            NDarray odds = np.exp(logitsArray);
            NDarray probs = odds / np.sum(odds);

            return new Tuple<dynamic, NDarray>(value, probs);
        }

        public float evaluateLeaf(Node leaf, float value, int done)
        {

            if (done == 0)
            {
                Tuple<dynamic, NDarray> tuple = get_preds(leaf.state);
                value = (float)tuple.Item1;
                NDarray probs = tuple.Item2;

                for (int i = 0; i <= leaf.state.AllowedActions.Count - 1; i++)
                {
                    int action = leaf.state.AllowedActions[i];
                    GameState newState = leaf.state.TakeAction(action);
                    Node node;
                    if (!this.mcts.tree.ContainsKey(newState.Id))
                    {
                        node = new Node(newState);
                        this.mcts.addNode(node);
                    }
                    else
                    {
                        node = this.mcts.tree[newState.Id];
                    }
                    Edge newEdge = new Edge(leaf, node, (float)probs[action], action);
                    leaf.edges.Add(action, newEdge);
                }
            }
            return value;
        }

        public Tuple<NDarray, NDarray> getAV(int tau)
        {
            List<int> piList = new List<int>();
            List<float> valuesList = new List<float>();
            for (int i = 0; i <= action_size - 1; i++)
            {
                piList.Add(0);
                valuesList.Add(0);
            }

            NDarray pi = np.array(piList.ToArray());
            NDarray values = np.array(valuesList.ToArray());

            foreach (KeyValuePair<int, Edge> kvp in this.mcts.root.edges)
            {
                int action = kvp.Key;
                Edge edge = kvp.Value;
                pi[action] = np.array(Math.Pow(edge.stats["N"], 1 / tau));
                values[action] = np.array(edge.stats["Q"]);
            }
            pi = pi / (np.sum(pi) * 1.0);
            return new Tuple<NDarray, NDarray>(pi, values);
        }
        public int chooseAction(NDarray pi, NDarray values, int tau)
        {
            int action;
            //if (tau == 0)
            {
                double[] piArray = (pi.GetData<double>());
                double max = piArray.Max();
                action = piArray.ToList().IndexOf(max);
            }
            return action;
        }

        public void buildMCTS(GameState state)
        {
            this.root = new Node(state);
            this.mcts = new MCTS(this.root, this.cpuct);
        }

        public void changeRootMCTS(GameState state)
        {
            this.mcts.root = this.mcts.tree[state.Id];
        }

    }
}
