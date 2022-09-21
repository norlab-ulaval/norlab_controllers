//
// Created by dominic on 9/21/22.
//

#ifndef NORLAB_CONTROLLERS_DIFFERENTIALORTHOGONALEXPONENTIAL_H
#define NORLAB_CONTROLLERS_DIFFERENTIALORTHOGONALEXPONENTIAL_H

#include <controllers/Controller.h>
#include <string>

namespace norlab_controllers{
    class DifferentialOrthogonalExponential: public Controller
    {
    public:
        DifferentialOrthogonalExponential(const std::string &configFilePath);
        virtual std::vector<float> computeCommandVector(const std::vector<float> &state);
    };
}



#endif //NORLAB_CONTROLLERS_DIFFERENTIALORTHOGONALEXPONENTIAL_H
