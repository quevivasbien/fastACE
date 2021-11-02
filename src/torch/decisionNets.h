#ifndef DECISION_NETS_H
#define DECISION_NETS_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "base.h"


// converts an Eigen::ArrayXd to a torch::Tensor
torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray);


struct OfferEncoder : torch::nn::Module {
	/**
	We want to be able to take information about Offers as input to a model,
	but we want to make it as easy as possible for individual nets to work with,
	so we train concurrently with other models an encoder,
	which takes many features and condenses them to some `encodingSize`

	stackSize is the number of offers to be processed at the same time (number to be compared simultaneously by a given agent)
	numFeatures is the number of features for each offer
		typically will have size numAgents (who is offering?) + numGoods (what goods are offered?) + 1 (price)
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
	torch::nn::Linear final = nullptr;

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
	numGoods is the number of goods in the simulation
	flatSize is the size of the net's hidden layers

	input shape is [batch size] x stackSize x offerEncodingSize
	output shape is [batch size] x stackSize

	Output is interpreted as the probability that the agent takes each offer in the stack if that offer is affordable
	*/
	PurchaseNet(
		int offerEncodingSize,
		int stackSize,
		int numUtilParams,
		int numGoods,
		int flatSize
	);

	torch::Tensor forward(
		torch::Tensor offerEncodings,
		torch::Tensor utilParams,
		torch::Tensor budget,
		torch::Tensor inventory
	);

	torch::nn::Linear flatten = nullptr;
	torch::nn::Linear flatForward1 = nullptr;
	torch::nn::Linear flatForward2 = nullptr;
	torch::nn::Linear flatForward3 = nullptr;
	torch::nn::Linear final = nullptr;
};


class NeuralDecisionMaker {
	// a wrapper for various decision nets
public:
	NeuralDecisionMaker(
		Economy* economy,
		std::shared_ptr<OfferEncoder> offerEncoder
	) : economy(economy), offerEncoder(offerEncoder) {
		update_encodedOffers();
	};
// private:
	Economy* economy;
	std::shared_ptr<OfferEncoder> offerEncoder;
	std::shared_ptr<PurchaseNet> purchaseNet;

	torch::Tensor encodedOffers;
	std::vector<std::shared_ptr<const Offer>> offers;

	void update_encodedOffers() {
		offers = economy->get_market();
		unsigned int numOffers = offers.size();

		// NOTE: if the simulation is going to add more agents later,
		// then the size of this should be set to MORE than totalAgents
		torch::Tensor offerers = torch::zeros(
			{numOffers, economy->get_totalAgents()}
		);
		torch::Tensor goods = torch::empty(
			{numOffers, economy->get_numGoods()}
		);
		torch::Tensor prices = torch::empty({numOffers, 1});

		for (int i = 0; i < numOffers; i++) {
			auto offer = offers[i];
			// set offerer
			offerers[i][economy->get_id_for_agent(offer->offerer)] = 1;
			// set goods
			goods[i] = eigenToTorch(offer->quantities).squeeze(-1);
			// set prices
			prices[i] = offer->price;
		}
		torch::Tensor inputFeatures = torch::cat({goods, prices, offerers}, 1);

		encodedOffers = offerEncoder->forward(inputFeatures);
	}

	// void get_purchase_proba(unsigned int offerIndex) {
	//
	// }

};


#endif
