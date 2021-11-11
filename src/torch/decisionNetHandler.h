#ifndef NEURAL_DECISION_MAKER_H
#define NEURAL_DECISION_MAKER_H

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include "decisionNets.h"
#include "base.h"

namespace neural {

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray);

Eigen::ArrayXd torchToEigen(torch::Tensor tensor);


torch::Tensor get_purchase_probas(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    const torch::Tensor& encodedOffers
);

torch::Tensor get_job_probas(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    const torch::Tensor& encodedJobOffers
);


struct DecisionNetHandler {
	// a wrapper for various decision nets
	DecisionNetHandler(
		Economy* economy,
		std::shared_ptr<OfferEncoder> offerEncoder,
        std::shared_ptr<OfferEncoder> jobOfferEncoder,
		std::shared_ptr<PurchaseNet> purchaseNet,
        std::shared_ptr<PurchaseNet> firmPurchaseNet,
        std::shared_ptr<PurchaseNet> laborSearchNet,
        std::shared_ptr<ConsumptionNet> consumptionNet
	);

    DecisionNetHandler(Economy* economy);

	Economy* economy;
	std::shared_ptr<OfferEncoder> offerEncoder;
    std::shared_ptr<OfferEncoder> jobOfferEncoder;
	std::shared_ptr<PurchaseNet> purchaseNet;
    std::shared_ptr<PurchaseNet> firmPurchaseNet;
    std::shared_ptr<PurchaseNet> laborSearchNet;
    std::shared_ptr<ConsumptionNet> consumptionNet;

	torch::Tensor encodedOffers;
    int numEncodedOffers;
	std::vector<std::shared_ptr<const Offer>> offers;

    torch::Tensor encodedJobOffers;
    int numEncodedJobOffers;
    std::vector<std::shared_ptr<const JobOffer>> jobOffers;

    int time = 0;

	void update_encodedOffers();

    void update_encodedJobOffers();

    void time_step();

    std::vector<Order<Offer>> create_offer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt
        torch::Tensor purchase_probas
    );

	std::vector<Order<Offer>> get_offers_to_request(
		const torch::Tensor& offerIndices, // dtype = kInt
		const Eigen::ArrayXd& utilParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
	);

    std::vector<Order<Offer>> firm_get_offers_to_request(
        const torch::Tensor& offerIndices, // dtype = kInt
		const Eigen::ArrayXd& prodFuncParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    std::vector<Order<JobOffer>> create_joboffer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt
        torch::Tensor job_probas
    );

    std::vector<Order<JobOffer>> get_joboffers_to_request(
        const torch::Tensor& offerIndices, // dtype = kInt
        const Eigen::ArrayXd& utilParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    Eigen::ArrayXd get_consumption_proportions(
        const Eigen::ArrayXd& utilParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

};

} // namespace neural

#endif
