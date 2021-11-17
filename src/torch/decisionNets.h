#ifndef DECISION_NETS_H
#define DECISION_NETS_H

#include <torch/torch.h>
#include <memory>

namespace neural {

void xavier_init(torch::nn::Module& module);

// create_linear just creates new linear layer and applies xavier_init
torch::nn::Linear create_linear(int in_features, int out_features);


struct OfferEncoder : torch::nn::Module {
	/**
	We want to be able to take information about Offers as input to a model,
	but we want to make it as easy as possible for individual nets to work with,
	so we train concurrently with other models an encoder,
	which takes many features and condenses them to some `encodingSize`

	stackSize is the number of offers to be processed at the same time (number to be compared simultaneously by a given agent)
	numFeatures is the number of features for each offer
		typically will have size numAgents (who is offering?) + numGoods (what goods are offered?) + 1 (price)
		for encoding JobOffers, will have size numAgents + 2 (offerer, labor, and wage)
	intermediateSize is size of hidden layers
	encodingSize is the size of the output (encoded) layer

	Takes inputs with shape [batch size] x stackSize x numFeatures
	Outputs shape [batch size] x stackSize x encodingSize
	*/
	OfferEncoder(
		int stackSize,
		int numFeatures,
		int intermediateSize,
		int encodingSize
	);

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Linear dimReduce = nullptr;
	torch::nn::Linear stackedForward1 = nullptr;
	torch::nn::Linear stackedForward2 = nullptr;
	torch::nn::Linear last = nullptr;

	int stackSize;
	int encodingSize;
};


struct PurchaseNet : torch::nn::Module {
	/**
	The goal of this net is to determine whether to take a goods offer
	We input information about the offer, as well as information about some other set of available offers
	Then output a probability, which is the proba of taking the offer, conditional upon it being affordable

	This net piggy-backs on OfferEncoder to get info about available offers.
	We also use the agent-in-question's utility params, budget, and current inventory as inputs
		(This should, theoretically, allow us to use the same PurchaseNet for all agents)

	offerEncodingSize is the encodingSize of the OfferEncoding this net get's its offer data from
	stackSize is the number of offers to be processed at the same time (number to be compared simultaneously by a given agent)
	numUtilParams is the number of parameters of the agents' utility functions to be included
	* NOTE this can also be used for firms' purchasing decisions, in which case,
		numUtilParams in the number of parameters of the firms' production function
	* NOTE can also be generalized to jobOffer acceptance decisions
	numGoods is the number of goods in the simulation
	flatSize is the size of the net's hidden layers
	*/
	PurchaseNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
	);

	torch::Tensor forward(
		// offerEncodings should be [batch size] x stackSize x offerEncodingSize
		// utilParams should be [batch size] x numUtilParams
		// budget should be [batch size] x 1
		// inventory should be [batch size] x numGoods
		// Output is interpreted as the probability that the agent takes each offer in the stack
			// if that offer is affordable
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor budget,
		torch::Tensor labor,
		torch::Tensor inventory
	);

	torch::nn::Linear flatten = nullptr;
	torch::nn::Linear flatForward1 = nullptr;
	torch::nn::Linear flatForward2 = nullptr;
	torch::nn::Linear flatForward3 = nullptr;
	torch::nn::Linear last = nullptr;

	int stackSize;
	int numUtilParams;
};


struct ConsumptionNet : torch::nn::Module {
	/**
	This net takes as an input utilParams and the current inventory, money, and labor of an agent,
	and returns alpha and beta params for Beta distribution over proportion of each good to consume
	output size will be numGoods x 2, where rows are mu and logsigma params for each good
	*/
	ConsumptionNet(
		int numUtilParams,
		int numGoods,
		int hiddenSize
	);

	torch::Tensor forward(
		torch::Tensor utilParams,
		torch::Tensor money,
		torch::Tensor labor,
		torch::Tensor inventory
	);

	torch::nn::Linear first = nullptr;
	torch::nn::Linear hidden1 = nullptr;
	torch::nn::Linear hidden2 = nullptr;
	torch::nn::Linear last = nullptr;

	int numUtilParams;
	int numGoods;
};


struct OfferNet : torch::nn::Module {
	/**
	This net takes as an inputs offerEncodings, utilParams and the current inventory, money, and labor of an agent,
	and returns (1) mu and sigma params for proportion of each good to sell,
	and (2) mu and logsigma params for per-good prices at which to sell

	implementation looks very much like that of PurchaseNet; input should have the same form
	output is a [batchSize] x numGoods x 4 tensor
	{prop_mu, prop_logsigma, price_mu, price_logsigma}
	*/
	OfferNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
	);

	torch::Tensor forward(
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor money,
		torch::Tensor labor,
		torch::Tensor inventory
	);

	torch::nn::Linear flatten = nullptr;
	torch::nn::Linear flatForward1 = nullptr;
	torch::nn::Linear flatForward2 = nullptr;
	torch::nn::Linear flatForward3a = nullptr;
	torch::nn::Linear flatForward3b = nullptr;
	torch::nn::Linear lasta = nullptr;
	torch::nn::Linear lastb = nullptr;

	int stackSize;
	int numUtilParams;
	int numGoods;
};


struct JobOfferNet : torch::nn::Module {
	/**
	Similar to OfferNet, but returns only a 4d vector
	{labor_mu, labor_logsigma, wage_per_labor_mu, wage_per_labor_logsigma}
	*/
	JobOfferNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
	);

	torch::Tensor forward(
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor money,
		torch::Tensor labor,
		torch::Tensor inventory
	);

	torch::nn::Linear flatten = nullptr;
	torch::nn::Linear flatForward1 = nullptr;
	torch::nn::Linear flatForward2 = nullptr;
	torch::nn::Linear flatForward3 = nullptr;
	torch::nn::Linear last = nullptr;

	int stackSize;
	int numUtilParams;
};


struct ValueNet : torch::nn::Module {
	/**
	Takes inputs representing state (the same inputs used to make purchasing decisions)
	and returns a single value representing how good it is to be in that state
	*/
	ValueNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int hiddenSize
	);

	torch::Tensor forward(
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor money,
		torch::Tensor labor,
		torch::Tensor inventory
	);

	torch::nn::Linear flatten = nullptr;
	torch::nn::Linear flatForward1 = nullptr;
	torch::nn::Linear flatForward2 = nullptr;
	torch::nn::Linear last = nullptr;

	int stackSize;
	int numUtilParams;
};


} // namespace neural

#endif
