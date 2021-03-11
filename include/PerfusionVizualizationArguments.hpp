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
    float endTime = 47844;
    float sweepTime = 5316;
    float sweepOffset = 2072.5;
    uint32_t sweepCount = 10;
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
    std::string staticReconstructionDir = "";
    // For visualization of static reconstruction points

    // Settings
    CLI::Option_group* og_settings = nullptr;
    // Time in seconds to define interval [0, cbf_time) for maximum computation for CBF
    float cbf_time = 5.0f;
    float lambda_rel = 0.2;
    /**
     * @brief File to store AIF.
     */
    std::string aifImageFile = "";
    std::string aifCsvFile = "";
    std::string basisImageFile = "";

protected:
    void addIntervalGroup();
    void addIntervalArgs();
    void addSweepArgs(bool includeSweepCount);
    void addGranularity();
    void addSecLength();
    void addSettingsGroup();
    void addSettingsArgs();
    void addVizualizationGroup();
    void addVizualizationArgs(bool staticReconstructionVisualization);
};
} // namespace CTL::util
