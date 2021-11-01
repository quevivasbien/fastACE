#include "neuralPersonDecisionMaker.h"

struct PersonDecisionMaker {
    // called by UtilMaxer::buy_goods()
    // looks at current goods on market and chooses bundle that maximizes utility
    // subject to restriction that total price is within budget
    virtual std::vector<Order<Offer>> choose_goods() = 0;
    // analogous to GoodChooser::choose_goods(), but for jobs
    // in default implementation does not take utility of labor into account
    // i.e. only tries to maximize wages
    virtual std::vector<Order<JobOffer>> choose_jobs() = 0;
    // Selects which goods in inventory should be consumed
    virtual Eigen::ArrayXd choose_goods_to_consume() = 0;

    PersonDecisionMaker(std::shared_ptr<UtilMaxer> parent);
    std::shared_ptr<UtilMaxer> parent;
};


struct NeuralPersonDecisionMaker : public PersonDecisionMaker {
	NeuralPersonDecisionMaker();
    NeuralPersonDecisionMaker(
    	std::shared_ptr<UtilMaxer> parent,
    	std::shared_ptr<torch::nn::Module> goodChooserNet;
    	std::shared_ptr<torch::nn::Module> jobChooserNet;
    	std::shared_ptr<torch::nn::Module> consumptionChooserNet;
	);

	virtual std::vector<Order<Offer>> choose_goods() override;

	virtual std::vector<Order<JobOffer>> choose_jobs() override;

	virtual Eigen::ArrayXd choose_goods_to_consume() override;
}