#ifndef NORLAB_CONTROLLERS_CONTROLLER_H
#define NORLAB_CONTROLLERS_CONTROLLER_H

#include "Path.h"
#include <string>
#include <vector>

namespace norlab_controllers
{
    class Controller
    {
    private:
        float commandRate;
        Path path;
    public:
        virtual std::vector<float> computeCommandVector(const std::vector<float> &state) = 0;
        std::vector<float> updatePath(const Path &newPath);
    };
}

#endif //NORLAB_CONTROLLERS_CONTROLLER_H
