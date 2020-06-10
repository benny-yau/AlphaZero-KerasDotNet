using ConnectFour;
using Keras.Models;
using Numpy;
using System.Collections.Generic;
using System.IO;

namespace AlphaZeroKerasDotNet
{
    public class Gen_Model
    {
        protected double reg_const;
        protected double learning_rate;
        protected List<int> input_dim;
        protected int output_dim;
        public Gen_Model(double reg_const, double learning_rate, List<int> input_dim, int output_dim)
        {
            this.reg_const = reg_const;
            this.learning_rate = learning_rate;
            this.input_dim = input_dim;
            this.output_dim = output_dim;
        }
    }

    public class Residual_CNN : Gen_Model
    {
        public BaseModel model;
        public Residual_CNN(double reg_const, double learning_rate, List<int> input_dim, int output_dim) : base(reg_const, learning_rate, input_dim, output_dim)
        {
        }
        public NDarray convertToModelInput(GameState state)
        {
            NDarray inputToModel = np.array(state.Binary.ToArray());
            inputToModel = np.reshape(inputToModel, this.input_dim.ToArray());
            return inputToModel;
        }

        public void ReadModel(int player_version)
        {
            model = Sequential.ModelFromJson(File.ReadAllText("player_NN.json"));
            model.LoadWeight("version" + player_version.ToString().PadLeft(4, '0') + ".h5");
        }
    }
}
