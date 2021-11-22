#include "decisionNets.h"

namespace neural {

void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}


OfferEncoder::OfferEncoder(
	int stackSize,
	int numFeatures,
	int intermediateSize,
	int encodingSize
) : stackSize(stackSize), encodingSize(encodingSize) {
	dimReduce = register_module(
		"dimReduce",
		torch::nn::Linear(numFeatures, intermediateSize)
	);
	stackedForward1 = register_module(
		"stackedForward1",
		torch::nn::Linear(intermediateSize, intermediateSize)
	);
	stackedForward2 = register_module(
		"stackedForward2",
		torch::nn::Linear(intermediateSize, intermediateSize)
	);
	last = register_module(
		"last",
		torch::nn::Linear(intermediateSize, encodingSize)
	);
	this->apply(xavier_init);
}

torch::Tensor OfferEncoder::forward(torch::Tensor x) {
	// todo: check that stack size is correct
	// first step is to reduce number of features
	// std::cout << "input: " << x << std::endl;
	x = torch::relu(dimReduce->forward(x));
	// next we go through a couple of linear layers
	x = x + torch::relu(stackedForward1->forward(x));
	x = x + torch::relu(stackedForward2->forward(x));
	// next we reduce to the encoding size and return
	x = torch::relu(last->forward(x));
	// std::cout << "output: " << x << std::endl;
	return x;
}


PurchaseNet::PurchaseNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize
) : numUtilParams(numUtilParams) {
	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	flatForward1 = register_module(
		"flatForward1",
		torch::nn::Linear(offerEncoder->stackSize + numUtilParams + numGoods + 2, hiddenSize)
	);
	flatForward2 = register_module(
		"flatForward2",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	flatForward3 = register_module(
		"flatForward3",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	last = register_module(
		"last",
		torch::nn::Linear(hiddenSize, offerEncoder->stackSize)
	);
	this->apply(xavier_init);

	// This is necessary in order to pass along the encoder's parameters
	// when doing backpropagation on the PurchaseNet
	this->offerEncoder = register_module("offerEncoder", offerEncoder);
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
) : numUtilParams(numUtilParams), numGoods(numGoods) {
    first = register_module(
		"first",
		torch::nn::Linear(numUtilParams + numGoods + 2, hiddenSize)
	);
    hidden1 = register_module(
		"hidden1",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
    hidden2 = register_module(
		"hidden2",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
    last = register_module(
		"last",
		torch::nn::Linear(hiddenSize, numGoods * 2)
	);
	this->apply(xavier_init);
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
    return last->forward(x).reshape({numGoods, 2});
}


OfferNet::OfferNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize
) : numUtilParams(numUtilParams), numGoods(numGoods) {
	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	flatForward1 = register_module(
		"flatForward1",
		torch::nn::Linear(offerEncoder->stackSize + numUtilParams + numGoods + 2, hiddenSize)
	);
	flatForward2 = register_module(
		"flatForward2",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	flatForward3a = register_module(
		"flatForward3a",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	flatForward3b = register_module(
		"flatForward3b",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	lasta = register_module(
		"lasta",
		torch::nn::Linear(hiddenSize, numGoods * 2)
	);
	lastb = register_module(
		"lastb",
		torch::nn::Linear(hiddenSize, numGoods * 2)
	);
	this->apply(xavier_init);

	this->offerEncoder = register_module("offerEncoder", offerEncoder);
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
	xa = lasta->forward(xa).reshape({numGoods, 2});
	xb = lastb->forward(xb).reshape({numGoods, 2});
	// return outputs in a stack, dim = numGoods x 4
	return torch::cat({xa, xb}, -1);
}


JobOfferNet::JobOfferNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize
) : numUtilParams(numUtilParams) {
	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	flatForward1 = register_module(
		"flatForward1",
		torch::nn::Linear(offerEncoder->stackSize + numUtilParams + numGoods + 2, hiddenSize)
	);
	flatForward2 = register_module(
		"flatForward2",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	flatForward3 = register_module(
		"flatForward3",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	last = register_module(
		"last",
		torch::nn::Linear(hiddenSize, 4)
	);
	this->apply(xavier_init);

	this->offerEncoder = offerEncoder;
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
	// std::cout << "after flatten: " << x << std::endl;
	// we can now add in the other features
	x = torch::cat({x, utilParams, money, labor, inventory}, -1);
	// std::cout << "after cat: " << x << std::endl;
	// last we do a few basic linear layers
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	x = x + torch::relu(flatForward3->forward(x));
	// std::cout << "after flatForwards: " << x << std::endl;
	// compute final outputs
	return last->forward(x);
}


ValueNet::ValueNet(
	std::shared_ptr<OfferEncoder> offerEncoder,
	std::shared_ptr<OfferEncoder> jobOfferEncoder,
	int numUtilParams,
	int numGoods,
	int hiddenSize
) : numUtilParams(numUtilParams) {
	offerFlatten = register_module(
		"offerFlatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	jobOfferFlatten = register_module(
		"jobOfferFlatten",
		torch::nn::Linear(jobOfferEncoder->encodingSize, 1)
	);
	flatForward1 = register_module(
		"flatForward1",
		torch::nn::Linear(
			offerEncoder->stackSize + jobOfferEncoder->stackSize + numUtilParams + numGoods + 2,
			hiddenSize
		)
	);
	flatForward2 = register_module(
		"flatForward2",
		torch::nn::Linear(hiddenSize, hiddenSize)
	);
	last = register_module(
		"last",
		torch::nn::Linear(hiddenSize, 1)
	);
	this->apply(xavier_init);

	this->offerEncoder = register_module("offerEncoder", offerEncoder);
	this->jobOfferEncoder = register_module("jobOfferEncoder", jobOfferEncoder);
}

torch::Tensor ValueNet::forward(
	const torch::Tensor& offerEncodings,
	const torch::Tensor& jobOfferEncodings,
	const torch::Tensor& utilParams,
	const torch::Tensor& money,
	const torch::Tensor& labor,
	const torch::Tensor& inventory
) {
	// how this works is old news by now...
	torch::Tensor offerX = torch::tanh(offerFlatten->forward(offerEncodings).squeeze(-1));
	torch::Tensor jobOfferX = torch::tanh(jobOfferFlatten->forward(jobOfferEncodings).squeeze(-1));
	torch::Tensor x = torch::cat({offerX, jobOfferX, utilParams, money, labor, inventory}, -1);
	x = torch::relu(flatForward1->forward(x));
	x = x + torch::relu(flatForward2->forward(x));
	return last->forward(x);
}

} // namespace neural
