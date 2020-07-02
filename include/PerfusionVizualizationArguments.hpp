#pragma once

#include "PROG/Arguments.hpp"
#include "rawop.h"

namespace CTL::util {

class PerfusionVizualizationArguments : public virtual Arguments
{
public:
    PerfusionVizualizationArguments(int argc, char* argv[], std::string appName);
    // Dimensions
    CLI::Option_group* og_interval = nullptr;
    /**
     * The first sweep and the last sweep should be identified with the ends of the interval.
     * startTime default is the end of the first sweep and endTime is the start of the last sweep
     * Data are from the experiments. Controls interval [ms].
     */
    float startTime = 4145;
    float endTime = 43699;
    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;
    // Length of one second in the units of the domain
    float secLength = 1000;

    CLI::Option_group* og_vizualization = nullptr;
    bool vizualize = false;
    bool showBasis = false;
    bool showAIF = false;
    bool stopAfterVizualization = false;
    bool stopAfterTTP = false;
    /*Default negative to show raw values.
     */
    float water_value = -0.027;
    /**
     * @brief File to store AIF.
     */
    std::string aifImageFile = "";
    std::string basisImageFile = "";

protected:

    void addIntervalGroup();
    void addIntervalArgs();
    void addVizualizationGroup();
    void addVizualizationArgs();
};
} // namespace CTL::util
