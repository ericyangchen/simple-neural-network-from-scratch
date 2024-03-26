# result
python train.py --dataset linear --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid
python train.py --dataset xor --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid

# discussion
## different learning rate
python different_lr.py --dataset linear --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid
python different_lr.py --dataset xor --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid

## different hidden size
python different_hidden_size.py --dataset linear --learning_rate 0.1 --model FullyConnectedNeuralNetwork --activation sigmoid --epoch 30000
python different_hidden_size.py --dataset xor --learning_rate 0.1 --model FullyConnectedNeuralNetwork --activation sigmoid

## without activation function
python without_activation.py --dataset linear --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation none
python without_activation.py --dataset xor --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation none


# extra
## different optimizer
python momentum_optimizer.py --dataset linear --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid --optimizer Momentum
python momentum_optimizer.py --dataset xor --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork --activation sigmoid --optimizer Momentum

## different activation
python different_activation.py --dataset linear --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork
python different_activation.py --dataset xor --learning_rate 0.1 --hidden_size 16 --model FullyConnectedNeuralNetwork
