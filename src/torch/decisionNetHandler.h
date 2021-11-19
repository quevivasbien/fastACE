#ifndef DECISION_NET_HANDLER_H
#define DECISION_NET_HANDLER_H

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include <mutex>
#include <utility>
#include "decisionNets.h"
#include "base.h"
#include "neuralEconomy.h"

namespace neural {

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray);

Eigen::ArrayXd torchToEigen(torch::Tensor tensor);

// params is [batchsize] x n x 2 tensor
// cols are {mu, sigma} for each of n obs
// returns pair where first value is n sampled values from normal dist
// and second value is log probas of those values
std::pair<torch::Tensor, torch::Tensor> sample_normal(torch::Tensor params);

// same as sample_normal, but applies sigmoid function to output values
std::pair<torch::Tensor, torch::Tensor> sample_sigmoidNormal(torch::Tensor params);

// same as sample_normal, but applies exp function to output values
std::pair<torch::Tensor, torch::Tensor> sample_logNormal(torch::Tensor params);


torch::Tensor get_purchase_probas(
    torch::Tensor offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    torch::Tensor encodedOffers
);

torch::Tensor get_job_probas(
    torch::Tensor offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    torch::Tensor encodedJobOffers
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
    std::vector<torch::Tensor> offerEncoderLogProba;
    std::shared_ptr<OfferEncoder> jobOfferEncoder;
    std::vector<torch::Tensor> jobOfferEncoderLogProba;
	std::shared_ptr<PurchaseNet> purchaseNet;
    std::vector<torch::Tensor> purchaseNetLogProba;
    std::shared_ptr<PurchaseNet> firmPurchaseNet;
    std::vector<torch::Tensor> firmPurchaseNetLogProba;
    std::shared_ptr<PurchaseNet> laborSearchNet;
    std::vector<torch::Tensor> laborSearchNetLogProba;
    std::shared_ptr<ConsumptionNet> consumptionNet;
    std::vector<torch::Tensor> consumptionNetLogProba;
    std::shared_ptr<ConsumptionNet> productionNet;
    std::vector<torch::Tensor> productionNetLogProba;
    std::shared_ptr<OfferNet> offerNet;
    std::vector<torch::Tensor> offerNetLogProba;
    std::shared_ptr<JobOfferNet> jobOfferNet;
    std::vector<torch::Tensor> jobOfferNetLogProba;

	torch::Tensor encodedOffers;
    int numEncodedOffers;
	std::vector<std::shared_ptr<const Offer>> offers;

    torch::Tensor encodedJobOffers;
    int numEncodedJobOffers;
    std::vector<std::shared_ptr<const JobOffer>> jobOffers;

    int time = -1;

    std::mutex myMutex;

	void update_encodedOffers();

    void update_encodedJobOffers();

    void push_back_logProbas();

    void time_step();

    torch::Tensor generate_offerIndices();
    torch::Tensor generate_jobOfferIndices();
    torch::Tensor firm_generate_offerIndices();
    torch::Tensor firm_generate_jobOfferIndices();

    std::vector<Order<Offer>> create_offer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        const torch::Tensor& purchase_probas,
        std::vector<torch::Tensor>& logProba
    );

	std::vector<Order<Offer>> get_offers_to_request(
        const torch::Tensor& offerIndices,
		const Eigen::ArrayXd& utilParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
	);

    std::vector<Order<Offer>> firm_get_offers_to_request(
        const torch::Tensor& offerIndices,
		const Eigen::ArrayXd& prodFuncParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    std::vector<Order<JobOffer>> create_joboffer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        const torch::Tensor& job_probas,
        std::vector<torch::Tensor>& logProba
    );

    std::vector<Order<JobOffer>> get_joboffers_to_request(
        const torch::Tensor& jobOfferIndices,
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

    torch::Tensor getEncodedOffersForOfferCreation(
        const torch::Tensor& offerIndices
    );

    torch::Tensor getEncodedOffersForJobOfferCreation(
        const torch::Tensor& offerIndices
    );

    std::pair<Eigen::ArrayXd, Eigen::ArrayXd> choose_offers(
        const torch::Tensor& offerIndices,
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    std::pair<double, double> choose_job_offers(
        const torch::Tensor& offerIndices,
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

};

} // namespace neural

#endif
