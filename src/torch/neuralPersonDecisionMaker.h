#ifndef NEURAL_PERSON_DECISION_MAKER_H
#define NEURAL_PERSON_DECISION_MAKER_H

#include <torch/torch.h>
#include "utilMaxer.h"
#include "neuralEconomy.h"

namespace neural {

struct NeuralPersonDecisionMaker : PersonDecisionMaker {

	NeuralPersonDecisionMaker(std::shared_ptr<DecisionNetHandler> guide);

	virtual std::vector<Order<Offer>> choose_goods() override;
	virtual std::vector<Order<JobOffer>> choose_jobs() override;
	virtual Eigen::ArrayXd choose_goods_to_consume() override;

	std::shared_ptr<DecisionNetHandler> guide;

protected:
	NeuralPersonDecisionMaker(
    	std::shared_ptr<UtilMaxer> parent,
    	std::shared_ptr<DecisionNetHandler> guide
	);

	void confirm_synchronized();
	void record_state_value();
	Eigen::ArrayXd get_utilParams() const;

	torch::Tensor myOfferIndices;
	torch::Tensor myJobOfferIndices;

	unsigned int time;
};

} // namespace neural

#endif
