#ifndef DECISION_NET_HANDLER_H
#define DECISION_NET_HANDLER_H

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include <mutex>
#include <utility>
#include "decisionNets.h"
#include "base.h"
# include "neuralEconomy.h"

namespace neural {

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray);

Eigen::ArrayXd torchToEigen(torch::Tensor tensor);


torch::Tensor get_purchase_probas(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    const torch::Tensor& encodedOffers
);

torch::Tensor get_job_probas(
    const torch::Tensor& offerIndices, // dtype = kInt64
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
		NeuralEconomy* economy,
		std::shared_ptr<OfferEncoder> offerEncoder,
        std::shared_ptr<OfferEncoder> jobOfferEncoder,
		std::shared_ptr<PurchaseNet> purchaseNet,
        std::shared_ptr<PurchaseNet> firmPurchaseNet,
        std::shared_ptr<PurchaseNet> laborSearchNet,
        std::shared_ptr<ConsumptionNet> consumptionNet,
        std::shared_ptr<ConsumptionNet> productionNet,
        std::shared_ptr<OfferNet> offerNet,
        std::shared_ptr<JobOfferNet> jobOfferNet
	);

    DecisionNetHandler(NeuralEconomy* economy);

	NeuralEconomy* economy;
	std::shared_ptr<OfferEncoder> offerEncoder;
    std::shared_ptr<OfferEncoder> jobOfferEncoder;
	std::shared_ptr<PurchaseNet> purchaseNet;
    std::shared_ptr<PurchaseNet> firmPurchaseNet;
    std::shared_ptr<PurchaseNet> laborSearchNet;
    std::shared_ptr<ConsumptionNet> consumptionNet;
    std::shared_ptr<ConsumptionNet> productionNet;
    std::shared_ptr<OfferNet> offerNet;
    std::shared_ptr<JobOfferNet> jobOfferNet;

	torch::Tensor encodedOffers;
    int numEncodedOffers;
	std::vector<std::shared_ptr<const Offer>> offers;

    torch::Tensor encodedJobOffers;
    int numEncodedJobOffers;
    std::vector<std::shared_ptr<const JobOffer>> jobOffers;

    int time = 0;

    std::mutex myMutex;

	void update_encodedOffers();

    void update_encodedJobOffers();

    void time_step();

    std::vector<Order<Offer>> create_offer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        torch::Tensor purchase_probas
    );

	std::vector<Order<Offer>> get_offers_to_request(
		const Eigen::ArrayXd& utilParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
	);

    std::vector<Order<Offer>> firm_get_offers_to_request(
		const Eigen::ArrayXd& prodFuncParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    std::vector<Order<JobOffer>> create_joboffer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        torch::Tensor job_probas
    );

    std::vector<Order<JobOffer>> get_joboffers_to_request(
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

    Eigen::ArrayXd get_production_proportions(
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    torch::Tensor getEncodedOffersForOfferCreation();

    std::pair<Eigen::ArrayXd, Eigen::ArrayXd> choose_offers(
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    std::pair<double, double> choose_job_offers(
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

};

} // namespace neural

#endif
