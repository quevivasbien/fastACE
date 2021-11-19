#ifndef NEURAL_FIRM_DECISION_MAKER_H
#define NEURAL_FIRM_DECISION_MAKER_H

#include <torch/torch.h>
#include "profitMaxer.h"
#include "decisionNetHandler.h"

namespace neural {

struct NeuralFirmDecisionMaker : FirmDecisionMaker {

    NeuralFirmDecisionMaker(std::shared_ptr<DecisionNetHandler> guide);

    virtual Eigen::ArrayXd choose_production_inputs() override;
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() override;
    virtual std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() override;

    void confirm_synchronized();
	Eigen::ArrayXd get_prodFuncParams() const;

	std::shared_ptr<DecisionNetHandler> guide;

protected:
    NeuralFirmDecisionMaker(
        std::shared_ptr<ProfitMaxer> parent,
        std::shared_ptr<DecisionNetHandler> guide
    );

    torch::Tensor myOfferIndices;
    torch::Tensor myJobOfferIndices;

    unsigned int time;
};

} // namespace neural

#endif
