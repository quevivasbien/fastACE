#include "decisionNets.h"

namespace neural {

void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

torch::nn::Linear create_linear(int in_features, int out_features) {
	torch::nn::Linear linear(in_features, out_features);
	linear->apply(xavier_init);
	return linear;
}


OfferEncoder::OfferEncoder(
	int stackSize,
	int numFeatures,
	int intermediateSize,
	int encodingSize
) : stackSize(stackSize), encodingSize(encodingSize) {
	dimReduce = create_linear(numFeatures, intermediateSize);
	stackedForward1 = create_linear(intermediateSize, intermediateSize);
	stackedForward2 = create_linear(intermediateSize, intermediateSize);
	last = create_linear(intermediateSize, encodingSize);
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
		int hiddenSize
) : stackSize(stackSize), numUtilParams(numUtilParams) {
	flatten = create_linear(offerEncodingSize, 1);
	flatForward1 = create_linear(stackSize + numUtilParams + numGoods + 2, hiddenSize);
	flatForward2 = create_linear(hiddenSize, hiddenSize);
	flatForward3 = create_linear(hiddenSize, hiddenSize);
	last = create_linear(hiddenSize, stackSize);
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
    first = create_linear(numUtilParams + numGoods + 2, hiddenSize);
    hidden1 = create_linear(hiddenSize, hiddenSize);
    hidden2 = create_linear(hiddenSize, hiddenSize);
    last = create_linear(hiddenSize, numGoods);
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


OfferNet::OfferNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
) : stackSize(stackSize), numUtilParams(numUtilParams) {
	flatten = create_linear(offerEncodingSize, 1);
	flatForward1 = create_linear(stackSize + numUtilParams + numGoods + 2, hiddenSize);
	flatForward2 = create_linear(hiddenSize, hiddenSize);
	flatForward3a = create_linear(hiddenSize, hiddenSize);
	flatForward3b = create_linear(hiddenSize, hiddenSize);
	lasta = create_linear(hiddenSize, numGoods);
	lastb = create_linear(hiddenSize, numGoods);
}

torch::Tensor OfferNet::forward(
		const torch::Tensor& offerEncodings,
		const torch::Tensor& utilParams,
		const torch::Tensor& money,
        const torch::Tensor& labor,
		const torch::Tensor& inventory
) {
	// first we get a single value for every element in the stack
	torch::Tensor x = torch::tanh(flatten->forward(offerEncodings).squeeze(-1));
	// we can now add in the other features
	x = torch::cat({x, utilParams, money, labor, inventory}, -1);
	// last we do a few basic linear layers
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	// here split off into determining quantity vector (a) and price vector (b)
	torch::Tensor xa = x + torch::relu(flatForward3a->forward(x));
	torch::Tensor xb = x + torch::relu(flatForward3b->forward(x));
	// compute final outputs
	xa = torch::sigmoid(lasta->forward(xa));
	xb = torch::exp(lastb->forward(xb));
	// return outputs in a stack, dim = numGoods x 2
	return torch::stack({xa, xb}, -1);
}


JobOfferNet::JobOfferNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
) : stackSize(stackSize), numUtilParams(numUtilParams) {
	flatten = create_linear(offerEncodingSize, 1);
	flatForward1 = create_linear(stackSize + numUtilParams + numGoods + 2, hiddenSize);
	flatForward2 = create_linear(hiddenSize, hiddenSize);
	flatForward3 = create_linear(hiddenSize, hiddenSize);
	last = create_linear(hiddenSize, 2);
}

torch::Tensor JobOfferNet::forward(
		const torch::Tensor& offerEncodings,
		const torch::Tensor& utilParams,
		const torch::Tensor& money,
        const torch::Tensor& labor,
		const torch::Tensor& inventory
) {
	// first we get a single value for every element in the stack
	torch::Tensor x = torch::tanh(flatten->forward(offerEncodings).squeeze(-1));
	// we can now add in the other features
	x = torch::cat({x, utilParams, money, labor, inventory}, -1);
	// last we do a few basic linear layers
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	x = x + torch::relu(flatForward3->forward(x));
	// compute final outputs
	return torch::exp(last->forward(x));
}

} // namespace neural
