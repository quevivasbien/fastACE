#ifndef NEURAL_PERSON_DECISION_MAKER_H
#define NEURAL_PERSON_DECISION_MAKER_H

#include "utilMaxer.h"
#include "decisionNets.h"


struct NeuralPersonDecisionMaker : public PersonDecisionMaker {
	NeuralPersonDecisionMaker();
    NeuralPersonDecisionMaker(
    	std::shared_ptr<UtilMaxer> parent,
    	std::shared_ptr<torch::nn::Module> offerEncoder,
    	std::shared_ptr<torch::nn::Module> goodChooserNet,
    	std::shared_ptr<torch::nn::Module> jobChooserNet,
    	std::shared_ptr<torch::nn::Module> consumptionChooserNet
	);

	virtual std::vector<Order<Offer>> choose_goods() override;

	virtual std::vector<Order<JobOffer>> choose_jobs() override;

	virtual Eigen::ArrayXd choose_goods_to_consume() override;
};

#endif
