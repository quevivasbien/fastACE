#include "decisionNets.h"

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray) {
    auto t = torch::empty({eigenArray.cols(),eigenArray.rows()});
    float* data = t.data_ptr<float>();

    Eigen::Map<Eigen::ArrayXf> arrayMap(data, t.size(1), t.size(0));
    arrayMap = eigenArray.cast<float>();
    return t.transpose(0, 1);
}


OfferEncoder::OfferEncoder(
	int stackSize,
	int numFeatures,
	int intermediateSize,
	int encodingSize
) : stackSize(stackSize), encodingSize(encodingSize) {
	dimReduce = torch::nn::Linear(numFeatures, intermediateSize);
	stackedForward1 = torch::nn::Linear(intermediateSize, intermediateSize);
	stackedForward2 = torch::nn::Linear(intermediateSize, intermediateSize);
	final = torch::nn::Linear(intermediateSize, encodingSize);
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
	x = torch::relu(final->forward(x));
	return x;
}


PurchaseNet::PurchaseNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int flatSize
) {
	flatten = torch::nn::Linear(offerEncodingSize + 1, 1);
	flatForward1 = torch::nn::Linear(stackSize + numUtilParams + numGoods + 1, flatSize);
	flatForward2 = torch::nn::Linear(flatSize, flatSize);
	flatForward3 = torch::nn::Linear(flatSize, flatSize);
	final = torch::nn::Linear(flatSize, stackSize);
	// todo: add better init for weights
}

torch::Tensor PurchaseNet::forward(
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor budget,
		torch::Tensor inventory
) {
	// first we get a single value for every element in the stack
	torch::Tensor x = torch::tanh(flatten->forward(x).squeeze(-1));
	// we can now add in the other features
	x = torch::cat({x, utilParams, budget, inventory}, -1);
	// last we do a few basic linear layers
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	x = x + torch::relu(flatForward3->forward(x));
	// finally, output 1d sigmoid
	return torch::sigmoid(final->forward(x));
}
