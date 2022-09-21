#include "Controller.h"

#include <iostream>

std::vector<float> norlab_controllers::Controller::updatePath(const Path &newPath)
{
    path = newPath;
}
