#ifndef NEURAL_ECONOMY_H
#define NEURAL_ECONOMY_H


#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include "neuralConstants.h"
#include "base.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "decisionNets.h"


namespace neural {


// forward declaration
struct DecisionNetHandler;


class NeuralEconomy : public Economy {
    /**
    This is an economy that is designed to interact with a DecisionNetHandler
    */
public:
    static std::shared_ptr<NeuralEconomy> init(
        std::vector<std::string> goods
    );
    static std::shared_ptr<NeuralEconomy> init(
        std::vector<std::string> goods,
        std::shared_ptr<DecisionNetHandler> handler
    );
    
    static std::shared_ptr<NeuralEconomy> init_dummy(unsigned int numGoods);

    std::weak_ptr<DecisionNetHandler> handler;

    virtual std::string get_typename() const;

    bool time_step_no_grad();

protected:
    NeuralEconomy(std::vector<std::string> goods);
};


// Defined in neuralPersonDecisionMaker.cpp

struct NeuralPersonDecisionMaker : PersonDecisionMaker {

	NeuralPersonDecisionMaker(std::weak_ptr<DecisionNetHandler> guide);

	virtual std::vector<Order<Offer>> choose_goods() override;
	virtual std::vector<Order<JobOffer>> choose_jobs() override;
	virtual Eigen::ArrayXd choose_goods_to_consume() override;

	std::weak_ptr<DecisionNetHandler> guide;

protected:
	NeuralPersonDecisionMaker(
    	std::weak_ptr<UtilMaxer> parent,
    	std::weak_ptr<DecisionNetHandler> guide
	);

	void confirm_synchronized();
	void record_state_value();
	Eigen::ArrayXd get_utilParams() const;

	torch::Tensor myOfferIndices;
	torch::Tensor myJobOfferIndices;

	Eigen::ArrayXd utilParams;

	unsigned int time = 0;
};


// Defined in neuralFirmDecisionMaker.cpp

struct NeuralFirmDecisionMaker : FirmDecisionMaker {

    NeuralFirmDecisionMaker(std::weak_ptr<DecisionNetHandler> guide);

    virtual Eigen::ArrayXd choose_production_inputs() override;
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() override;
    virtual std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() override;

	std::weak_ptr<DecisionNetHandler> guide;

protected:
    NeuralFirmDecisionMaker(
        std::weak_ptr<ProfitMaxer> parent,
        std::weak_ptr<DecisionNetHandler> guide
    );

    void confirm_synchronized();
	void record_state_value();
	void record_profit();
	Eigen::ArrayXd get_prodFuncParams() const;

    torch::Tensor myOfferIndices;
    torch::Tensor myJobOfferIndices;

    Eigen::ArrayXd prodFuncParams;

	double last_money;
    unsigned int time = 0;
};


// FROM HERE ON ARE DEFINED IN decisionNetHandler.cpp

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray);

Eigen::ArrayXd torchToEigen(torch::Tensor tensor);

// params is [batchsize] x n x 2 tensor
// cols are {mu, logSigma} for each of n obs (note *log* sigma; sigma = exp(logSigma))
// returns pair where first value is n sampled values from normal dist
// and second value is log probas of those values
std::pair<torch::Tensor, torch::Tensor> sample_normal(const torch::Tensor& params);

// same as sample_normal, but applies sigmoid function to output values
std::pair<torch::Tensor, torch::Tensor> sample_logitNormal(const torch::Tensor& params);

// same as sample_normal, but applies exp function to output values
std::pair<torch::Tensor, torch::Tensor> sample_logNormal(const torch::Tensor& params);


torch::Tensor get_purchase_probas(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    const std::shared_ptr<PurchaseNet>& purchaseNet,
    const torch::Tensor& encodedOffers
);

torch::Tensor get_job_probas(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    const std::shared_ptr<PurchaseNet>& laborSearchNet,
    const torch::Tensor& encodedJobOffers
);


struct DecisionNetHandler {
	// a wrapper for various decision nets
	DecisionNetHandler(
		std::shared_ptr<NeuralEconomy> economy,
		std::shared_ptr<OfferEncoder> offerEncoder,
        std::shared_ptr<OfferEncoder> jobOfferEncoder,
		std::shared_ptr<PurchaseNet> purchaseNet,
        std::shared_ptr<PurchaseNet> firmPurchaseNet,
        std::shared_ptr<PurchaseNet> laborSearchNet,
        std::shared_ptr<ConsumptionNet> consumptionNet,
        std::shared_ptr<ConsumptionNet> productionNet,
        std::shared_ptr<OfferNet> offerNet,
        std::shared_ptr<JobOfferNet> jobOfferNet,
        std::shared_ptr<ValueNet> valueNet,
        std::shared_ptr<ValueNet> firmValueNet
	);

    DecisionNetHandler(
        std::shared_ptr<NeuralEconomy> economy,
        unsigned int stackSize,
        unsigned int encodingSize,
        unsigned int hiddenSize,
        unsigned int nHidden,
        unsigned int nHiddenSmall
    );

    DecisionNetHandler(std::shared_ptr<NeuralEconomy> economy);

    using MapTensor = std::unordered_map<Agent*, torch::Tensor>;
    using VecMapTensor = std::vector<MapTensor>;

	std::shared_ptr<NeuralEconomy> economy;

	std::shared_ptr<OfferEncoder> offerEncoder;
    std::shared_ptr<OfferEncoder> jobOfferEncoder;
	std::shared_ptr<PurchaseNet> purchaseNet;
    std::shared_ptr<PurchaseNet> firmPurchaseNet;
    std::shared_ptr<PurchaseNet> laborSearchNet;
    std::shared_ptr<ConsumptionNet> consumptionNet;
    std::shared_ptr<ConsumptionNet> productionNet;
    std::shared_ptr<OfferNet> offerNet;
    std::shared_ptr<JobOfferNet> jobOfferNet;
    std::shared_ptr<ValueNet> valueNet;
    std::shared_ptr<ValueNet> firmValueNet;

    VecMapTensor purchaseNetLogProba;
    VecMapTensor firmPurchaseNetLogProba;
    VecMapTensor laborSearchNetLogProba;
    VecMapTensor consumptionNetLogProba;
    VecMapTensor productionNetLogProba;
    VecMapTensor offerNetLogProba;
    VecMapTensor jobOfferNetLogProba;

    VecMapTensor values;
    VecMapTensor rewards;

	torch::Tensor encodedOffers;
    int numEncodedOffers;
	std::vector<std::weak_ptr<const Offer>> offers;

    torch::Tensor encodedJobOffers;
    int numEncodedJobOffers;
    std::vector<std::weak_ptr<const JobOffer>> jobOffers;

    int time = -1;

    std::mutex myMutex;
    std::mutex purchaseNetMutex;
    std::mutex firmPurchaseNetMutex;
    std::mutex laborSearchNetMutex;
    std::mutex consumptionNetMutex;
    std::mutex productionNetMutex;
    std::mutex offerNetMutex;
    std::mutex jobOfferNetMutex;
    std::mutex valueNetMutex;
    std::mutex firmValueNetMutex;

	void update_encodedOffers();

    void update_encodedJobOffers();

    void push_back_memory();

    void time_step();

    void synchronize_time(const std::shared_ptr<Agent>& caller);

    void reset(std::shared_ptr<NeuralEconomy> newEconomy);

    torch::Tensor generate_offerIndices();
    torch::Tensor generate_jobOfferIndices();
    torch::Tensor firm_generate_offerIndices();
    torch::Tensor firm_generate_jobOfferIndices();

    std::pair<std::vector<Order<Offer>>, torch::Tensor> create_offer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        const torch::Tensor& purchase_probas
    );

	std::vector<Order<Offer>> get_offers_to_request(
        Agent* caller,
        const torch::Tensor& offerIndices,
		const Eigen::ArrayXd& utilParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
	);

    std::vector<Order<Offer>> firm_get_offers_to_request(
        Agent* caller,
        const torch::Tensor& offerIndices,
		const Eigen::ArrayXd& prodFuncParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    std::pair<std::vector<Order<JobOffer>>, torch::Tensor> create_joboffer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt64
        const torch::Tensor& job_probas
    );

    std::vector<Order<JobOffer>> get_joboffers_to_request(
        Agent* caller,
        const torch::Tensor& jobOfferIndices,
        const Eigen::ArrayXd& utilParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    Eigen::ArrayXd get_consumption_proportions(
        Agent* caller,
        const Eigen::ArrayXd& utilParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    Eigen::ArrayXd get_production_proportions(
        Agent* caller,
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    torch::Tensor getEncodedOffersFromIndices(
        const torch::Tensor& offerIndices
    );

    torch::Tensor getEncodedJobOffersFromIndices(
        const torch::Tensor& offerIndices
    );

    std::pair<Eigen::ArrayXd, Eigen::ArrayXd> choose_offers(
        Agent* caller,
        const torch::Tensor& offerIndices,
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    std::pair<double, double> choose_job_offers(
        Agent* caller,
        const torch::Tensor& offerIndices,
        const Eigen::ArrayXd& prodFuncParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    );

    void record_value(
        Agent* caller,
        const torch::Tensor& offerIndices,
        const torch::Tensor& jobOfferIndices,
		const Eigen::ArrayXd& utilParams,
		double money,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    void firm_record_value(
        Agent* caller,
        const torch::Tensor& offerIndices,
        const torch::Tensor& jobOfferIndices,
		const Eigen::ArrayXd& prodFuncParams,
		double money,
        double labor,
		const Eigen::ArrayXd& inventory
    );

    void record_reward(
        const std::shared_ptr<Agent>& caller,
        double reward
    );

    void record_reward(
        const std::shared_ptr<Agent>& caller,
        double reward,
        int offset
    );

    void save_models();
    void load_models();

};

} // namespace neural

#endif
