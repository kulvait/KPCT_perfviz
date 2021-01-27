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

void PerfusionVizualizationArguments::addIntervalArgs(bool sweepParameters)
{
    addIntervalGroup();
    std::string description;
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
    if(sweepParameters)
    {
        description = io::xprintf("Sweep time between the starts of two consecutive acquisitions "
                                  "in miliseconds. [defaults to %f]",
                                  sweepTime);
        og_interval->add_option("--sweep-time", sweepTime, description)
            ->check(CLI::Range(0.0, 100000.0));
        description = io::xprintf("Offset at the beginning and at the end of aquisition in "
                                  "seconds/1000. . [defaults to %f]",
                                  sweepOffset);
        og_interval->add_option("--sweep-offset", sweepOffset, description)
            ->check(CLI::Range(0.0, 100000.0));
        description = io::xprintf("Number of sweeps. [defaults to %d]", sweepCount);
        og_interval->add_option("--sweep-count", sweepCount, description)->check(CLI::Range(2, 20));
    }
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

void PerfusionVizualizationArguments::addVizualizationArgs(bool staticReconstructionVisualization)
{
    addVizualizationGroup();
    og_vizualization->add_flag("-v,--vizualize", vizualize, "Vizualization.");
    og_vizualization->add_option(
        "--water-value", water_value,
        io::xprintf("If the AIF vizualization should be in HU, use this water_value. For C-Arm "
                    "devices we usually set it to 0.027. Negative value imply use of raw values, "
                    "defaults to %f",
                    water_value));
    og_vizualization->add_flag("--show-basis", showBasis, "Show basis.");
    og_vizualization->add_flag("--show-aif", showAIF, "Show AIF.");
    og_vizualization->add_option("--store-aif", aifImageFile, "Store AIF into image file.");
    og_vizualization->add_option("--store-basis", basisImageFile, "Store basis into image file.");
    if(staticReconstructionVisualization)
    {
        og_vizualization->add_option("--static-reconstruction-dir", staticReconstructionDir,
                                     "Specify directory of the static reconstruction coeficients "
                                     "to visualize them along model based reconstruction.");
    }
}

void PerfusionVizualizationArguments::addSettingsGroup()
{
    if(getRegisteredOptionGroup("settings") == nullptr)
    {
        registerOptionGroup(
            "settings",
            cliApp->add_option_group("Settings of perfusion computation algorithm.", "Settings."));
    }
    og_settings = getRegisteredOptionGroup("settings");
}

void PerfusionVizualizationArguments::addSettingsArgs()
{
    addSettingsGroup();
    og_settings->add_option("--cbf-time", cbf_time,
                            io::xprintf("Time in seconds to specify interval [0, cbf_time) in "
                                        "which we compute maximum for CBF, defaults to %f",
                                        cbf_time));
    og_settings->add_flag("--lambda-rel", lambda_rel,
                          io::xprintf("Tikhonov relative lambda for Tikhonov stabilization of "
                                      "deconvolution operator derived from AIF, defaults to %f",
                                      lambda_rel));
}

} // namespace CTL::util
