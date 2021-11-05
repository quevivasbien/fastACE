#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include "decisionNets.h"
#include "base.h"


torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray, bool transpose) {
    auto t = torch::empty({eigenArray.cols(), eigenArray.rows()});
    float* data = t.data_ptr<float>();

    Eigen::Map<Eigen::ArrayXf> arrayMap(data, t.size(1), t.size(0));
    arrayMap = eigenArray.cast<float>();

    // t.requires_grad_(true);

    if (transpose) {
        return t;
    }
    else {
        return t.transpose(0, 1);
    }
}

torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray) {
    return eigenToTorch(eigenArray, false);
}


torch::Tensor get_purchase_probas(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    const torch::Tensor& encodedOffers
) {
    // get encoded offers
    auto offerEncodings = encodedOffers.index_select(0, offerIndices);

    // transform other inputs to torch Tensors
    auto utilParams_ = eigenToTorch(utilParams).squeeze(-1);
    auto budget_ = torch::tensor({budget});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory).squeeze(-1);

    // plug into purchaseNet to get probas
    return purchaseNet->forward(offerEncodings, utilParams_, budget_, labor_, inventory_);
}

torch::Tensor get_job_probas(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    const torch::Tensor& encodedJobOffers
) {
    // get encoded offers
    auto jobOfferEncodings = encodedJobOffers.index_select(0, offerIndices);

    // transform other inputs to torch Tensors
    auto utilParams_ = eigenToTorch(utilParams).squeeze(-1);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory).squeeze(-1);

    // plug into purchaseNet to get probas
    return laborSearchNet->forward(jobOfferEncodings, utilParams_, money_, labor_, inventory_);
}


struct NeuralDecisionMaker {
	// a wrapper for various decision nets
	NeuralDecisionMaker(
		Economy* economy,
		std::shared_ptr<OfferEncoder> offerEncoder,
        std::shared_ptr<OfferEncoder> jobOfferEncoder,
		std::shared_ptr<PurchaseNet> purchaseNet,
        std::shared_ptr<PurchaseNet> firmPurchaseNet,
        std::shared_ptr<PurchaseNet> laborSearchNet
	) : economy(economy),
        offerEncoder(offerEncoder),
        jobOfferEncoder(jobOfferEncoder),
        purchaseNet(purchaseNet),
        firmPurchaseNet(firmPurchaseNet),
        laborSearchNet(laborSearchNet)
    {
		update_encodedOffers();
        update_encodedJobOffers();
	};

	Economy* economy;
	std::shared_ptr<OfferEncoder> offerEncoder;
    std::shared_ptr<OfferEncoder> jobOfferEncoder;
	std::shared_ptr<PurchaseNet> purchaseNet;
    std::shared_ptr<PurchaseNet> firmPurchaseNet;
    std::shared_ptr<PurchaseNet> laborSearchNet;

	torch::Tensor encodedOffers;
    int numEncodedOffers;
	std::vector<std::shared_ptr<const Offer>> offers;

    torch::Tensor encodedJobOffers;
    int numEncodedJobOffers;
    std::vector<std::shared_ptr<const JobOffer>> jobOffers;

    int time = 0;

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
		auto inputFeatures = torch::cat({goods, prices, offerers}, 1);

		encodedOffers = offerEncoder->forward(inputFeatures);
        numEncodedOffers = numOffers;
	}

    void update_encodedJobOffers() {
        jobOffers = economy->get_jobMarket();
        unsigned int numOffers = jobOffers.size();

        // NOTE: this is inefficient, since only firms make job offers...
        torch::Tensor offerers = torch::zeros({numOffers, economy->get_totalAgents()});
        torch::Tensor labors = torch::empty({numOffers, 1});
        torch::Tensor wages = torch::empty({numOffers, 1});

        for (int i = 0; i < numOffers; i++) {
            auto jobOffer = jobOffers[i];
            offerers[i][economy->get_id_for_agent(jobOffer->offerer)] = 1;
            labors[i] = jobOffer->labor;
            wages[i] = jobOffer->wage;
        }
        auto inputFeatures = torch::cat({labors, wages, offerers}, 1);

        encodedJobOffers = jobOfferEncoder->forward(inputFeatures);
        numEncodedJobOffers = numOffers;
    }

    void time_step() {
        update_encodedOffers();
        update_encodedJobOffers();
        time++;
    }

    std::vector<std::shared_ptr<const Offer>> create_offer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt
        torch::Tensor purchase_probas
    ) {
        auto to_purchase = (torch::rand(purchase_probas.sizes()) < purchase_probas);

        std::vector<std::shared_ptr<const Offer>> toRequest;
		for (int i = 0; i < to_purchase.size(0); i++) {
			if (to_purchase[i].item<bool>()) {
				toRequest.push_back(offers[offerIndices[i].item<int>()]);
			}
		}
		return toRequest;
    }

	std::vector<std::shared_ptr<const Offer>> get_offers_to_request(
		const torch::Tensor& offerIndices, // dtype = kInt
		const Eigen::ArrayXd& utilParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
	) {
		auto probas = get_purchase_probas(
            offerIndices, utilParams, budget, labor, inventory, purchaseNet, encodedOffers
        );

		return create_offer_requests(offerIndices, probas);
	}

    std::vector<std::shared_ptr<const Offer>> firm_get_offers_to_request(
        const torch::Tensor& offerIndices, // dtype = kInt
		const Eigen::ArrayXd& prodFuncParams,
		double budget,
        double labor,
		const Eigen::ArrayXd& inventory
    ) {
        auto probas = get_purchase_probas(
            offerIndices, prodFuncParams, budget, labor, inventory, firmPurchaseNet, encodedOffers
        );

		return create_offer_requests(offerIndices, probas);
    }

    std::vector<std::shared_ptr<const JobOffer>> create_joboffer_requests(
        const torch::Tensor& offerIndices, // dtype = kInt
        torch::Tensor job_probas
    ) {
        auto to_take = (torch::rand(job_probas.sizes()) < job_probas);

        std::vector<std::shared_ptr<const JobOffer>> toRequest;
        for (int i = 0; i < to_take.size(0); i++) {
			if (to_take[i].item<bool>()) {
				toRequest.push_back(jobOffers[offerIndices[i].item<int>()]);
			}
		}
		return toRequest;
    }

    std::vector<std::shared_ptr<const JobOffer>> get_joboffers_to_request(
        const torch::Tensor& offerIndices, // dtype = kInt
        const Eigen::ArrayXd& utilParams,
        double money,
        double labor,
        const Eigen::ArrayXd& inventory
    ) {
        auto probas = get_job_probas(
            offerIndices, utilParams, money, labor, inventory, laborSearchNet, encodedJobOffers
        );

        return create_joboffer_requests(offerIndices, probas);
    }



};
