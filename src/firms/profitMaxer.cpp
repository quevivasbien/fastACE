#include "profitMaxer.h"

ProfitMaxer::ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner, unsigned int outputIndex) :
    Firm(economy, owner),
    prodFunc(
        std::make_shared<VToVFromVToS>(
            std::make_shared<CobbDouglas>(economy->get_numGoods() + 1),
            economy->get_numGoods(),
            outputIndex
        )
    ),
    goodProducer(std::make_shared<GreedyGoodProducer>(this)),
    goodSeller(std::make_shared<GreedyGoodSeller>(this)),
    laborFinder(std::make_shared<GreedyLaborFinder>(this))
{}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::shared_ptr<Agent> owner,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<GoodProducer> goodProducer,
    std::shared_ptr<GoodSeller> goodSeller,
    std::shared_ptr<LaborFinder> laborFinder
) : Firm(economy, owner),
    prodFunc(prodFunc),
    goodProducer(goodProducer),
    goodSeller(goodSeller),
    laborFinder(laborFinder)
{
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
    // these assertions are to make sure that the helper classes don't get assigned to more than one ProfitMaxer
    assert(goodProducer->parent == nullptr);
    assert(goodSeller->parent == nullptr);
    assert(laborFinder->parent == nullptr);
    goodProducer->parent = this;
    goodSeller->parent = this;
    laborFinder->parent = this;
}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<GoodProducer> goodProducer,
    std::shared_ptr<GoodSeller> goodSeller,
    std::shared_ptr<LaborFinder> laborFinder
) : Firm(economy, owners, inventory, money),
    prodFunc(prodFunc),
    goodProducer(goodProducer),
    goodSeller(goodSeller),
    laborFinder(laborFinder)
{
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
    // these assertions are to make sure that the helper classes don't get assigned to more than one ProfitMaxer
    assert(goodProducer->parent == nullptr);
    assert(goodSeller->parent == nullptr);
    assert(laborFinder->parent == nullptr);
    goodProducer->parent = this;
    goodSeller->parent = this;
    laborFinder->parent = this;
}


GoodProducer::GoodProducer(ProfitMaxer* parent) : parent(parent) {}
GoodSeller::GoodSeller(ProfitMaxer* parent) : parent(parent) {}
LaborFinder::LaborFinder(ProfitMaxer* parent) : parent(parent) {}

Eigen::ArrayXd ProfitMaxer::f(const Eigen::ArrayXd& quantities) {
    return prodFunc->f(quantities);
}


void ProfitMaxer::produce() {
    Eigen::ArrayXd inputs = goodProducer->choose_production_inputs();
    inventory += (f(inputs) - inputs(Eigen::seq(1, Eigen::placeholders::last)));
}

void ProfitMaxer::sell_goods() {
    auto offers = goodSeller->choose_good_offers();
    for (auto offer : offers) {
        post_offer(offer);
    }
}

void ProfitMaxer::search_for_laborers() {
    auto jobOffers = laborFinder->choose_job_offers();
    for (auto offer : jobOffers) {
        post_jobOffer(offer);
    }
}


GreedyGoodProducer::GreedyGoodProducer(ProfitMaxer* parent) : GoodProducer(parent) {}

Eigen::ArrayXd GreedyGoodProducer::choose_production_inputs() {
    Eigen::ArrayXd out(parent->get_economy()->get_numGoods() + 1);
    out << parent->get_laborHired(), parent->get_inventory();
    return out;
}


GreedyGoodSeller::GreedyGoodSeller(ProfitMaxer* parent) : GoodSeller(parent) {}

std::vector<std::shared_ptr<Offer>> GreedyGoodSeller::choose_good_offers() {
    // dummy value for now
    std::vector<std::shared_ptr<Offer>> offers;
    return offers;
}


GreedyLaborFinder::GreedyLaborFinder(ProfitMaxer* parent) : LaborFinder(parent) {}

std::vector<std::shared_ptr<JobOffer>> GreedyLaborFinder::choose_job_offers() {
    // dummy value for now
    std::vector<std::shared_ptr<JobOffer>> offers;
    return offers;
}
