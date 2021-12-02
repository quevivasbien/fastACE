#include "neuralEconomy.h"

namespace neural {

NeuralEconomy::NeuralEconomy(
    std::vector<std::string> goods
) : Economy(goods) {}


std::shared_ptr<NeuralEconomy> NeuralEconomy::init(
    std::vector<std::string> goods,
    std::shared_ptr<DecisionNetHandler> handler
) {
    std::shared_ptr<NeuralEconomy> self(
        new NeuralEconomy(goods)
    );
    handler->economy = self;
    self->handler = handler;
    return self;
}

std::shared_ptr<NeuralEconomy> NeuralEconomy::init(
    std::vector<std::string> goods
) {
    // leaves handler uninitialized
    return std::shared_ptr<NeuralEconomy>(
        new NeuralEconomy(goods)
    );
}

std::shared_ptr<NeuralEconomy> NeuralEconomy::init_dummy(unsigned int numGoods) {
    std::vector<std::string> goods(numGoods);
    for (unsigned int i = 0; i < numGoods; i++) {
        goods[i] = "good" + std::to_string(i);
    }
    return std::shared_ptr<NeuralEconomy>(new NeuralEconomy(goods));
}

std::string NeuralEconomy::get_typename() const {
    return "NeuralEconomy";
}

} // namespace neural
