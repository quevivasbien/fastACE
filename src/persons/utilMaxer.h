#ifndef UTILMAXER_H
#define UTILMAXER_H

#include "base.h"
#include "constants.h"
#include "vecToScalar.h"
// #include "solve.h"


class UtilMaxer;

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


class UtilMaxer : public Person {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    template <typename ... Args>
    static std::shared_ptr<UtilMaxer> init(Args&& ... args) {
        auto utilMaxer = create<UtilMaxer>(std::forward<Args>(args) ...);
        utilMaxer->init_decisionMaker();
        return utilMaxer;
    }

    double u(double labor, const Eigen::ArrayXd& quantities);  // alias for utilFunc.f
    double u(const Eigen::ArrayXd& quantities);  // implicitly inputs labor = laborSupplied

    std::shared_ptr<const VecToScalar> get_utilFunc() const;

    virtual std::string get_typename() const override;

protected:
    UtilMaxer(Economy* economy);
    UtilMaxer(
        Economy* economy,
        std::shared_ptr<VecToScalar> utilFunc,
        std::shared_ptr<PersonDecisionMaker> decisionMaker
    );
    UtilMaxer(
        Economy* economy,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToScalar> utilFunc,
        std::shared_ptr<PersonDecisionMaker> decisionMaker
    );
    void init_decisionMaker();

    // utilFunc should be a VecToScalar with numInputs == economy->numGoods + 1
    // the first input is assumed to be 1 - labor (i.e. leisure), rest of inputs are goods quantities
    std::shared_ptr<VecToScalar> utilFunc;
    std::shared_ptr<PersonDecisionMaker> decisionMaker;

    // calls gooChooser->choose_goods(), then requests those goods
    virtual void buy_goods() override;

    // calls jobChooser->choose_jobs(), then requests those jobs
    virtual void search_for_jobs() override;

    // calls goodConsumer->choose_goods_to_consume(), then removes those goods from inventory
    virtual void consume_goods() override;
};


struct BasicPersonDecisionMaker : PersonDecisionMaker {
    BasicPersonDecisionMaker();
    BasicPersonDecisionMaker(std::shared_ptr<UtilMaxer> parent);

    virtual std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<Order<JobOffer>> choose_jobs() override;
    virtual Eigen::ArrayXd choose_goods_to_consume() override;
};


#endif
