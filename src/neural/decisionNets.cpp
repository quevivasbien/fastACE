#include <assert.h>
#include "decisionNets.h"

namespace neural {

void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

void perturb_layer(torch::nn::Linear& linear, double pct) {
    torch::NoGradGuard noGrad;
	// figure out what the standard deviation should be
	int in;
	int out;
	if (linear->weight.ndimension() == 2) {
		in = linear->weight.size(1);
		out = linear->weight.size(0);
	}
	else {
		in = linear->weight.size(1) * linear->weight[0][0].numel();
		out = linear->weight.size(0) * linear->weight[0][0].numel();
	}
	const auto xavier_var = 2.0 / (in + out);

	// we want to add noise but keep the xavier standard dev
	// var(aX + bY) = a^2 var(X) + b^2 var(Y) = xavier_var
	// here var(X) = xavier_var and Y is std normal noise, var(Y) = 1
	// b^2 = xavier_var (1 - a^2)
	// 1 - a^2 = pct, so noise should be sqrt(xavier_var * pct) * N(0, 1)
	auto noise = std::sqrt(xavier_var * pct) * torch::randn(linear->weight.sizes());
	linear->weight = std::sqrt(1 - pct) * linear->weight + noise;
}


OfferEncoder::OfferEncoder(
	int stackSize,
	int numFeatures,
	int hiddenSize,
	int numHidden,
	int encodingSize
) : stackSize(stackSize), numHidden(numHidden), encodingSize(encodingSize) {
	dimReduce = register_module(
		"dimReduce",
		torch::nn::Linear(numFeatures, hiddenSize)
	);
	hidden.reserve(numHidden);
	for (int i = 0; i < numHidden; i++) {
		hidden.push_back(
			register_module(
				"hidden" + std::to_string(i),
				torch::nn::Linear(hiddenSize, hiddenSize)
			)
		);
	}
	last = register_module(
		"last",
		torch::nn::Linear(hiddenSize, encodingSize)
	);
	this->apply(xavier_init);
}

torch::Tensor OfferEncoder::forward(torch::Tensor x) {
	// todo: check that stack size is correct
	// first step is to reduce number of features
	x = torch::tanh(dimReduce->forward(x));
	// next we go through fully hidden linear layers
	for (int i = 0; i < numHidden; i++) {
		// use residual connections whenever possible to help with trainability
		x = x + torch::tanh(hidden[i]->forward(x));
	}
	// finally we reduce to the encoding size and return
	return torch::tanh(last->forward(x));
}

void OfferEncoder::perturb_weights(double pct) {
	perturb_layer(dimReduce, pct);
	for (auto h : hidden) {
		perturb_layer(h, pct);
	}
	perturb_layer(last, pct);
}


PurchaseNet::PurchaseNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize,
		int numHidden
) : numUtilParams(numUtilParams), numHidden(numHidden) {
	
	assert(numHidden >= 1);

	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	int numFeatures = offerEncoder->stackSize + numUtilParams + numGoods + 2;
	hidden.reserve(numHidden);
	for (int i = 0; i < numHidden; i++) {
		int inputSize = (i > 0) ? hiddenSize : numFeatures;
		hidden.push_back(
			register_module(
				"hidden" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize)
			)
		);
	}
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
	x = torch::tanh(hidden[0]->forward(x));
	for (int i = 1; i < numHidden; i++) {
		// use residual connections whenever possible to help with trainability
		x = x + torch::tanh(hidden[i]->forward(x));
	}
	// lastly, output 1d sigmoid
	return torch::sigmoid(last->forward(x));
}

void PurchaseNet::perturb_weights(double pct) {
	perturb_layer(flatten, pct);
	for (auto h : hidden) {
		perturb_layer(h, pct);
	}
	perturb_layer(last, pct);
}


ConsumptionNet::ConsumptionNet(
    int numUtilParams,
    int numGoods,
    int hiddenSize,
	int numHidden
) : numUtilParams(numUtilParams), numGoods(numGoods), numHidden(numHidden) {
    first = register_module(
		"first",
		torch::nn::Linear(numUtilParams + numGoods + 2, hiddenSize)
	);
	hidden.reserve(numHidden);
	for (int i = 0; i < numHidden; i++) {
		hidden.push_back(
			register_module(
				"hidden" + std::to_string(i),
				torch::nn::Linear(hiddenSize, hiddenSize)
			)
		);
	}
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
    x = torch::tanh(first->forward(x));
	for (int i = 0; i < numHidden; i++) {
		// use residual connections whenever possible to help with trainability
		x = x + torch::tanh(hidden[i]->forward(x));
	}
    return last->forward(x).reshape({numGoods, 2});
}

void ConsumptionNet::perturb_weights(double pct) {
	perturb_layer(first, pct);
	for (auto h : hidden) {
		perturb_layer(h, pct);
	}
	perturb_layer(last, pct);
}


OfferNet::OfferNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize_firstStage,
		int hiddenSize_secondStage,
		int numHidden_firstStage,
		int numHidden_secondStage
) : numUtilParams(numUtilParams),
	numGoods(numGoods),
	numHidden_firstStage(numHidden_firstStage),
	numHidden_secondStage(numHidden_secondStage)
{
	assert(numHidden_firstStage >= 1);
	assert(numHidden_secondStage >= 1);

	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	
	int numFeatures = offerEncoder->stackSize + numUtilParams + numGoods + 2;
	hidden_firstStage.reserve(numHidden_firstStage);
	for (int i = 0; i < numHidden_firstStage; i++) {
		int inputSize = (i > 0) ? hiddenSize_firstStage : numFeatures;
		hidden_firstStage.push_back(
			register_module(
				"hidden_firstStage" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize_firstStage)
			)
		);
	}

	hidden_secondStage_a.reserve(numHidden_secondStage);
	hidden_secondStage_b.reserve(numHidden_secondStage);
	for (int i = 0; i < numHidden_secondStage; i++) {
		int inputSize = (i > 0) ? hiddenSize_secondStage : hiddenSize_firstStage;
		hidden_secondStage_a.push_back(
			register_module(
				"hidden_secondStage_a" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize_secondStage)
			)
		);
		hidden_secondStage_b.push_back(
			register_module(
				"hidden_secondStage_b" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize_secondStage)
			)
		);
	}
	last_a = register_module(
		"last_a",
		torch::nn::Linear(hiddenSize_secondStage, numGoods * 2)
	);
	last_b = register_module(
		"last_b",
		torch::nn::Linear(hiddenSize_secondStage, numGoods * 2)
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
	// now a few basic linear layers
	x = torch::tanh(hidden_firstStage[0]->forward(x));
	for (int i = 1; i < numHidden_firstStage; i++) {
		// use residual connections whenever possible to help with trainability
		x = x + torch::tanh(hidden_firstStage[i]->forward(x));
	}
	// here split off into determining quantity vector (a) and price vector (b)
	torch::Tensor x_a = x + torch::tanh(hidden_secondStage_a[0]->forward(x));
	torch::Tensor x_b = x + torch::tanh(hidden_secondStage_b[0]->forward(x));
	for (int i = 1; i < numHidden_secondStage; i++) {
		x_a = x_a + torch::tanh(hidden_secondStage_a[i]->forward(x_a));
		x_b = x_b + torch::tanh(hidden_secondStage_b[i]->forward(x_b));
	}
	// compute final outputs
	x_a = last_a->forward(x_a).reshape({numGoods, 2});
	x_b = last_b->forward(x_b).reshape({numGoods, 2});
	// return outputs in a stack, dim = numGoods x 4
	return torch::cat({x_a, x_b}, -1);
}

void OfferNet::perturb_weights(double pct) {
	perturb_layer(flatten, pct);
	for (auto h : hidden_firstStage) {
		perturb_layer(h, pct);
	}
	for (auto h : hidden_secondStage_a) {
		perturb_layer(h, pct);
	}
	for (auto h : hidden_secondStage_b) {
		perturb_layer(h, pct);
	}
	perturb_layer(last_a, pct);
	perturb_layer(last_b, pct);
}


JobOfferNet::JobOfferNet(
		std::shared_ptr<OfferEncoder> offerEncoder,
		int numUtilParams,
		int numGoods,
		int hiddenSize,
		int numHidden
) : numUtilParams(numUtilParams), numHidden(numHidden) {

	assert(numHidden >= 1);

	flatten = register_module(
		"flatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);

	int numFeatures = offerEncoder->stackSize + numUtilParams + numGoods + 2;
	hidden.reserve(numHidden);
	for (int i = 0; i < numHidden; i++) {
		int inputSize = (i > 0) ? hiddenSize : numFeatures;
		hidden.push_back(
			register_module(
				"hidden" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize)
			)
		);
	}

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
	// we can now add in the other features
	x = torch::cat({x, utilParams, money, labor, inventory}, -1);
	// last we do a few basic linear layers
	x = torch::tanh(hidden[0]->forward(x));
	for (int i = 1; i < numHidden; i++) {
		// use residual connections whenever possible to help with trainability
		x = x + torch::tanh(hidden[i]->forward(x));
	}
	// compute final outputs
	return last->forward(x);
}

void JobOfferNet::perturb_weights(double pct) {
	perturb_layer(flatten, pct);
	for (auto h : hidden) {
		perturb_layer(h, pct);
	}
	perturb_layer(last, pct);
}


ValueNet::ValueNet(
	std::shared_ptr<OfferEncoder> offerEncoder,
	std::shared_ptr<OfferEncoder> jobOfferEncoder,
	int numUtilParams,
	int numGoods,
	int hiddenSize,
	int numHidden
) : numUtilParams(numUtilParams), numHidden(numHidden) {
	
	assert(numHidden >= 1);

	offerFlatten = register_module(
		"offerFlatten",
		torch::nn::Linear(offerEncoder->encodingSize, 1)
	);
	jobOfferFlatten = register_module(
		"jobOfferFlatten",
		torch::nn::Linear(jobOfferEncoder->encodingSize, 1)
	);

	int numFeatures = offerEncoder->stackSize + jobOfferEncoder->stackSize + numUtilParams + numGoods + 2;
	hidden.reserve(numHidden);
	for (int i = 0; i < numHidden; i++) {
		int inputSize = (i > 0) ? hiddenSize : numFeatures;
		hidden.push_back(
			register_module(
				"hidden" + std::to_string(i),
				torch::nn::Linear(inputSize, hiddenSize)
			)
		);
	}

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
	x = torch::tanh(hidden[0]->forward(x));
	for (int i = 1; i < numHidden; i++) {
		x = x + torch::tanh(hidden[i]->forward(x));
	}
	return last->forward(x);
}

void ValueNet::perturb_weights(double pct) {
	perturb_layer(offerFlatten, pct);
	perturb_layer(jobOfferFlatten, pct);
	for (auto h : hidden) {
		perturb_layer(h, pct);
	}
	perturb_layer(last, pct);
}

} // namespace neural
