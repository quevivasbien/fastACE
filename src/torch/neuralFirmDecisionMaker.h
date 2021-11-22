#ifndef NEURAL_FIRM_DECISION_MAKER_H
#define NEURAL_FIRM_DECISION_MAKER_H

#include <torch/torch.h>
#include "profitMaxer.h"
#include "neuralEconomy.h"

namespace neural {

struct NeuralFirmDecisionMaker : FirmDecisionMaker {

    NeuralFirmDecisionMaker(std::shared_ptr<DecisionNetHandler> guide);

    virtual Eigen::ArrayXd choose_production_inputs() override;
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() override;
    virtual std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() override;

    void confirm_synchronized();
	void record_state_value();
	void record_profit();
	Eigen::ArrayXd get_prodFuncParams() const;

	std::shared_ptr<DecisionNetHandler> guide;

protected:
    NeuralFirmDecisionMaker(
        std::shared_ptr<ProfitMaxer> parent,
        std::shared_ptr<DecisionNetHandler> guide
    );

    torch::Tensor myOfferIndices;
    torch::Tensor myJobOfferIndices;

    Eigen::ArrayXd prodFuncParams;

	double last_money;
    unsigned int time = 0;
};

} // namespace neural

#endif
