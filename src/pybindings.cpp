#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include "pybindings.h"


neural::CustomScenarioParams create_scenario_params(
    unsigned int numPeople,
    unsigned int numFirms
) {
    return neural::CustomScenarioParams(numPeople, numFirms);
}

neural::TrainingParams create_training_params(
) {
    return neural::TrainingParams();
}

void print_offer_info(
    const std::vector<std::weak_ptr<const Offer>>& offers,
    const std::vector<std::string>& goods
) {
    unsigned int numGoods = goods.size();
    std::vector<double> sumPrices(numGoods);
    std::vector<unsigned int> counts(numGoods);
    for (auto offer_ : offers) {
        auto offer = offer_.lock();
        for (unsigned int i = 0; i < numGoods; i++) {
            if (offer->quantities(i) > 0) {
                sumPrices[i] += (offer->quantities(i) / offer->price);
                counts[i]++;
            }
        }
    }
    for (unsigned int i = 0; i < numGoods; i++) {
        if (counts[i] > 0) {
            std::cout << goods[i] << ": Avg. price = " << sumPrices[i] / counts[i] << " (num. offers = " << counts[i] << ")\n";
        }
        else {
            std::cout << goods[i] << ": Avg. price = NA (num. offers = 0)\n";
        }
    }
}

void print_jobOffer_info(
    std::vector<std::weak_ptr<const JobOffer>>& jobOffers
) {
    double sumWage = 0.0;
    for (auto jobOffer_ : jobOffers) {
        auto jobOffer = jobOffer_.lock();
        sumWage += jobOffer->wage / jobOffer->labor;
    }
    std::cout << "Avg. wage per unit of labor = " << sumWage / jobOffers.size() << " (num. offers = " << jobOffers.size() << ")\n";
}

void print_info(const neural::NeuralEconomy& economy) {
    std::cout << "Time = " << economy.get_time() << ":\n";
    auto offers = economy.get_market();
    if (offers.size() > 0) {
        auto goods = economy.get_goods();
        print_offer_info(offers, goods);
    }
    else {
        std::cout << "[No offers]\n";
    }

    auto jobOffers = economy.get_jobMarket();
    if (jobOffers.size() > 0) {
        print_jobOffer_info(jobOffers);
    }
    else {
        std::cout << "[No job offers]\n";
    }
}


void run(
    neural::CustomScenarioParams scenarioParams,
    neural::TrainingParams trainingParams
) {
    std::shared_ptr<neural::CustomScenario> scenario = neural::create_scenario(scenarioParams, trainingParams);
    scenario->handler->load_models();
    auto economy = std::static_pointer_cast<neural::NeuralEconomy>(scenario->setup());
    for (unsigned int t = 0; t < trainingParams.episodeLength; t++) {
        economy->time_step_no_grad();
        print_info(*economy);
    }
}


void train(
    double* output,
    const neural::CustomScenarioParams* scenarioParams,
    neural::TrainingParams* trainingParams,
    bool fromPretrained,
    double perturbationSize
) {
    std::vector<double> losses = (
        (!fromPretrained) ?
        neural::train(
            *scenarioParams,
            *trainingParams
        )
        :
        neural::train_from_pretrained(
            *scenarioParams,
            *trainingParams,
            perturbationSize
        )
    );
    
    std::copy(losses.begin(), losses.end(), output);
}
