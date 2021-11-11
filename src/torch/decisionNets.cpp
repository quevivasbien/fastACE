#include "decisionNets.h"

namespace neural {

OfferEncoder::OfferEncoder(
	int stackSize,
	int numFeatures,
	int intermediateSize,
	int encodingSize
) : stackSize(stackSize), encodingSize(encodingSize) {
	dimReduce = torch::nn::Linear(numFeatures, intermediateSize);
	stackedForward1 = torch::nn::Linear(intermediateSize, intermediateSize);
	stackedForward2 = torch::nn::Linear(intermediateSize, intermediateSize);
	last = torch::nn::Linear(intermediateSize, encodingSize);
	// todo: add better init for weights
}

torch::Tensor OfferEncoder::forward(torch::Tensor x) {
	// todo: check that stack size is correct
	// first step is to reduce number of features
	x = torch::relu(dimReduce->forward(x));
	// next we go through a couple of linear layers
	x = x + torch::relu(stackedForward1->forward(x));
	x = x + torch::relu(stackedForward2->forward(x));
	// next we reduce to the encoding size and return
	x = torch::relu(last->forward(x));
	return x;
}


PurchaseNet::PurchaseNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int flatSize
) : stackSize(stackSize), numUtilParams(numUtilParams) {
	flatten = torch::nn::Linear(offerEncodingSize, 1);
	flatForward1 = torch::nn::Linear(stackSize + numUtilParams + numGoods + 2, flatSize);
	flatForward2 = torch::nn::Linear(flatSize, flatSize);
	flatForward3 = torch::nn::Linear(flatSize, flatSize);
	last = torch::nn::Linear(flatSize, stackSize);
	// todo: add better init for weights
}

torch::Tensor PurchaseNet::forward(
		const torch::Tensor& offerEncodings,
		const torch::Tensor& utilParams,
		const torch::Tensor& budget,
        const torch::Tensor& labor,
		const torch::Tensor& inventory
) {
	// first we get a single value for every element in the stack
	torch::Tensor x = torch::tanh(flatten->forward(offerEncodings).squeeze(-1));
	// we can now add in the other features
	x = torch::cat({x, utilParams, budget, labor, inventory}, -1);
	// last we do a few basic linear layers
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	x = x + torch::relu(flatForward3->forward(x));
	// lastly, output 1d sigmoid
	return torch::sigmoid(last->forward(x));
}


ConsumptionNet::ConsumptionNet(
    int numUtilParams,
    int numGoods,
    int hiddenSize
) : numUtilParams(numUtilParams) {
    first = torch::nn::Linear(numUtilParams + numGoods + 2, hiddenSize);
    hidden1 = torch::nn::Linear(hiddenSize, hiddenSize);
    hidden2 = torch::nn::Linear(hiddenSize, hiddenSize);
    last = torch::nn::Linear(hiddenSize, numGoods);
}

torch::Tensor ConsumptionNet::forward(
    const torch::Tensor& utilParams,
    const torch::Tensor& money,
    const torch::Tensor& labor,
    const torch::Tensor& inventory
) {
    torch::Tensor x = torch::cat({utilParams, money, labor, inventory}, -1);
    x = torch::relu(first->forward(x));
    x = x + torch::relu(hidden1->forward(x));
    x = x + torch::relu(hidden2->forward(x));
    return torch::sigmoid(last->forward(x));
}

} // namespace neural
