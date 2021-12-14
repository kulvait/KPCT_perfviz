// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "gitversion/version.h"

#include "PROG/Program.hpp"
//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "CSVWriter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "matplotlibcpp.h"
#include "stringFormatter.h"
#include "utils/EngineerSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

namespace plt = matplotlibcpp;

using namespace KCT;
using namespace KCT::util;

/// Arguments of the main function.
class Args : public ArgumentsThreading, public PerfusionVizualizationArguments
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , PerfusionVizualizationArguments(argc, argv, prgName){};

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> fittedCoefficients;

    /// Sampled function values in a DEN format to use as basis.
    std::string sampledBasis;

    /**
     * The first sweep and the last sweep should be identified with the ends of the interval.
     * startTime default is the end of the first sweep and endTime is the start of the last sweep
     * Data are from the experiments. Controls interval [ms].
     */
    float startTime = 4145, endTime = 43699;

    /// Coordinates of arthery input function
    uint32_t ifx, ify, ifz;
    uint32_t dimx, dimy, dimz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    uint32_t basisCapacity;

    bool allowNegativeValues;
};

void Args::defineArguments()
{
    cliApp->add_option("ifx", ifx, "Voxel x coordinate of the arthery input function.")->required();
    cliApp->add_option("ify", ify, "Voxel y coordinate of the arthery input function.")->required();
    cliApp->add_option("ifz", ifz, "Voxel z coordinate of the arthery input function.")->required();
    cliApp
        ->add_option("output_folder", outputFolder,
                     "Folder to which output data after the linear regression, specify - for no "
                     "computation of perfusion parameters.")
        ->required();
    cliApp
        ->add_option("sampled_basis", sampledBasis,
                     "Sampled basis functions to be used for perfusion parameters computation in a "
                     "DEN file, sampling is along the x axis and there is sizez sampled base "
                     "functions that should correspond to fittedCoefficients size.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("fitted_coeficients", fittedCoefficients,
                     "Coeficients of the basis functions fited by the algorithm. Orderred in the "
                     "same order as the basis is sampled in a DEN file.")
        ->required()
        ->check(CLI::ExistingFile);

    cliApp
        ->add_option("-i,--start-time", startTime,
                     "Start of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 4117, 247*16.6].")
        ->check(CLI::Range(0.0, 100000.0));
    cliApp
        ->add_option("-e,--end-time", endTime,
                     "End of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 56000, duration of 9 sweeps].")
        ->check(CLI::Range(0.0, 100000.0));
    cliApp
        ->add_option("-g,--granularity", granularity,
                     "Granularity of the time is number of time points to which time interval is "
                     "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    cliApp
        ->add_option("-c,--sec-length", secLength,
                     "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    addIntervalArgs();
    addVizualizationArgs(true);
    addSettingsArgs();
    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
}

int Args::postParse()
{
    std::string err;
    if(outputFolder.compare("-") == 0)
    {
        stopAfterVizualization = true;
        LOGI << "No directory specified so that after visualization program ends.";
    } else
    {
        if(!io::isDirectory(outputFolder))
        {
            err = io::xprintf("The path %s does not encode a valid directory!",
                              outputFolder.c_str());
            LOGE << err;
            return -1;
        }
    }
    if(fittedCoefficients.size() <= 1)
    {
        err = io::xprintf("There shall be at least two fitted coefficients!");
        LOGE << err;
        return -1;
    }
    io::DenFileInfo di(fittedCoefficients[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    if(ifx >= dimx || ify >= dimy || ifz >= dimz)
    {
        err = io::xprintf("Coordinates of the AIF must be within volume dimensions!");
        LOGE << err;
        return -1;
    }
    if(!(startTime < endTime))
    {
        err = io::xprintf("Start time %f must preceed end time %f.", startTime, endTime);
        LOGE << err;
        return -1;
    }
    if(secLength == 0.0)
    {
        err = io::xprintf("Length of the second is not positive!");
        LOGE << err;
        return -1;
    }
    if(vizualize)
    {
        showBasis = true;
        showAIF = true;
    }
    if(showBasis || showAIF)
    {
        vizualize = true;
    }
    io::DenFileInfo db(sampledBasis);
    basisCapacity = di.dimz();
    if(basisCapacity < fittedCoefficients.size())
    {
        err = io::xprintf("Fitted coefficients size %d is greater than the number of functions %d "
                          "in sampled basis",
                          fittedCoefficients.size(), basisCapacity);
        LOGE << err;
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    std::string prgInfo
        = "Visualization of perfusion parameters CT based on fiting of engineered basis.";
    if(version::MODIFIED_SINCE_COMMIT == true)
    {
        prgInfo = io::xprintf("%s Dirty commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    } else
    {
        prgInfo = io::xprintf("%s Git commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    }
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    std::shared_ptr<util::EngineerSeriesEvaluator> concentration
        = std::make_shared<util::EngineerSeriesEvaluator>(ARG.sampledBasis, ARG.fittedCoefficients,
                                                          ARG.startTime, ARG.endTime);
    // Vizualization
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
    float aifZeroValue = concentration->valueAt_intervalStart(ARG.ifx, ARG.ify, ARG.ifz);
    concentration->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(ARG.granularity, aif, convolutionMatrix);
    if(ARG.vizualize || !ARG.aifImageFile.empty() || !ARG.aifCsvFile.empty()
       || !ARG.basisImageFile.empty())
    {
        if(!ARG.vizualize)
        {
            plt::backend("Agg");
        }
        if(ARG.showBasis || !ARG.basisImageFile.empty())
        {
            util::StepFunction b(ARG.sampledBasis, ARG.fittedCoefficients.size(), ARG.startTime,
                                 ARG.endTime);
            if(ARG.showBasis)
            {
                b.plotFunctions();
            }
            if(!ARG.basisImageFile.empty())
            {
                b.storeFunctions(ARG.basisImageFile);
            }
        }
        plt::title(io::xprintf("Time attenuation curve, TST Engineer, x=%d, y=%d, z=%d.", ARG.ifx,
                               ARG.ify, ARG.ifz));
        plt::xlabel("Time [s]");
        if(ARG.water_value > 0)
        {
            plt::ylabel("Attenuation [HU]");
        } else
        {
            plt::ylabel("Attenuation");
        }
        std::vector<double> taxis;
        float* _taxis = new float[ARG.granularity];
        concentration->timeDiscretization(ARG.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            float v = aif[i] + aifZeroValue;
            if(ARG.water_value > 0)
            {
                float hu = 1000 * (v / ARG.water_value - 1.0);
                plotme.push_back(hu);
            } else
            {
                plotme.push_back(v);
            }
            taxis.push_back(_taxis[i] / ARG.secLength);
        }
        plt::plot(taxis, plotme);
        if(!ARG.aifImageFile.empty())
        {
            plt::save(ARG.aifImageFile);
        }
        if(ARG.showAIF)
        {
            plt::show();
        }
        if(!ARG.aifCsvFile.empty())
        {
            io::CSVWriter csv(ARG.aifCsvFile, "\t", true);
            csv.writeLine(io::xprintf("perfviz-engineer generated AIF (x,y,z)=(%d, %d, %d) with "
                                      "baseSize %d, startTime=%f and endTime=%f",
                                      ARG.ifx, ARG.ify, ARG.ifz, ARG.fittedCoefficients.size(),
                                      ARG.startTime, ARG.endTime));
            csv.writeVector(io::xprintf("time_engineer%d_interp", ARG.fittedCoefficients.size()),
                            taxis);
            csv.writeVector(io::xprintf("value_engineer%d_interp", ARG.fittedCoefficients.size()),
                            plotme);
            csv.close();
        }
        delete[] _taxis;
    }
    if(ARG.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, ARG.granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", ARG.outputFolder.c_str()), ARG.dimx, ARG.dimy, ARG.dimz);
    util::TimeSeriesDiscretizer tsd(concentration, ARG.dimx, ARG.dimy, ARG.dimz, ARG.startTime,
                                    ARG.endTime, ARG.secLength, ARG.threads);
    LOGD << "TTP computation.";
    tsd.computeTTP(ARG.granularity, ttp_w, aif);
    if(!ARG.stopAfterTTP)
    {
        LOGD << "CBV, CBF and MTT computation.";
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBF.den", ARG.outputFolder.c_str()), ARG.dimx, ARG.dimy, ARG.dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBV.den", ARG.outputFolder.c_str()), ARG.dimx, ARG.dimy, ARG.dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/MTT.den", ARG.outputFolder.c_str()), ARG.dimx, ARG.dimy, ARG.dimz);
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w,
                                       ARG.cbf_time);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
