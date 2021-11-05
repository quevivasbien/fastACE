#ifndef NEURAL_PERSON_DECISION_MAKER_H
#define NEURAL_PERSON_DECISION_MAKER_H

#include "utilMaxer.h"
#include "neuralDecisionMaker.h"


struct NeuralPersonDecisionMaker : public PersonDecisionMaker {
	NeuralPersonDecisionMaker();
    NeuralPersonDecisionMaker(
    	std::shared_ptr<UtilMaxer> parent,
    	std::shared_ptr<NeuralDecisionMaker> guide
	);

	virtual std::vector<Order<Offer>> choose_goods() override;

	virtual std::vector<Order<JobOffer>> choose_jobs() override;

	virtual Eigen::ArrayXd choose_goods_to_consume() override;

	void check_guide_is_current();
	Eigen::ArrayXd get_utilParams() const;

	std::shared_ptr<NeuralDecisionMaker> guide;
};

#endif
