#include "PerfusionVizualizationArguments.hpp"

namespace CTL::util {

PerfusionVizualizationArguments::PerfusionVizualizationArguments(int argc,
                                                                 char* argv[],
                                                                 std::string appName)
    : Arguments(argc, argv, appName){};
void PerfusionVizualizationArguments::addIntervalGroup()
{
    if(getRegisteredOptionGroup("interval") == nullptr)
    {
        registerOptionGroup(
            "interval",
            cliApp->add_option_group(
                "Interval specification",
                "Specification of the time interval parameters and its discretization."));
    }
    og_interval = getRegisteredOptionGroup("interval");
}

void PerfusionVizualizationArguments::addIntervalArgs()
{
    addIntervalGroup();
    og_interval
        ->add_option("-i,--start-time", startTime,
                     "Start of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 4117, 247*16.6].")
        ->check(CLI::Range(0.0, 100000.0));
    og_interval
        ->add_option("-e,--end-time", endTime,
                     "End of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 56000, duration of 9 sweeps].")
        ->check(CLI::Range(0.0, 100000.0));
    og_interval
        ->add_option("-c,--sec-length", secLength,
                     "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    og_interval
        ->add_option("-g,--granularity", granularity,
                     "Granularity of the time is number of time points to which time interval is "
                     "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
}

void PerfusionVizualizationArguments::addVizualizationGroup()
{
    if(getRegisteredOptionGroup("vizualization") == nullptr)
    {
        registerOptionGroup("vizualization",
                            cliApp->add_option_group(
                                "Vizualization configuration.",
                                "Configure output of basis and AIF to images and vizualization"));
    }
    og_vizualization = getRegisteredOptionGroup("vizualization");
}

void PerfusionVizualizationArguments::addVizualizationArgs()
{
    addVizualizationGroup();
    og_vizualization->add_flag("-v,--vizualize", vizualize, "Vizualization.");
    og_vizualization->add_option("--water-value", water_value,
                                 "If the AIF vizualization should be in HU, use this water_value. For C-Arm devices we usually set it to 0.027.");
    og_vizualization->add_flag("--show-basis", showBasis, "Show basis.");
    og_vizualization->add_flag("--show-aif", showAIF, "Show AIF.");
    og_vizualization->add_option("--store-aif", aifImageFile, "Store AIF into image file.");
    og_vizualization->add_option("--store-basis", basisImageFile, "Store basis into image file.");
}
} // namespace CTL::util
