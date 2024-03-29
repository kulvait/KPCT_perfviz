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
#include "FUN/FourierSeries.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "matplotlibcpp.h"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/FourierSeriesEvaluator.hpp"
#include "utils/ReconstructedSeriesEvaluator.hpp"
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

    /// Coordinates of arthery input function
    uint32_t ifx, ify, ifz;
    uint32_t dimx, dimy, dimz;

    // Specify basis
    bool halfPeriodicFunctions = false;

    bool allowNegativeValues = false;
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
        ->add_option("fitted_coeficients", fittedCoefficients,
                     "Fourier coeficients fited by the algorithm. Orderred from the first "
                     "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    addIntervalArgs();
    addSweepArgs(true);
    addVizualizationArgs(true);
    addSettingsArgs();
    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    cliApp->add_flag("--half-periodic-functions", halfPeriodicFunctions,
                     "Use Fourier basis and include half periodic functions.");
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
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    std::string prgInfo = "Visualization of perfusion parameters based on Fourier coefficients.";
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
    LOGI << io::xprintf("Start time is %f and end time is %f", ARG.startTime, ARG.endTime);
    uint32_t baseSize = ARG.fittedCoefficients.size();
    std::shared_ptr<util::FourierSeriesEvaluator> concentration
        = std::make_shared<util::FourierSeriesEvaluator>(ARG.fittedCoefficients, ARG.startTime,
                                                         ARG.endTime, !ARG.allowNegativeValues,
                                                         ARG.halfPeriodicFunctions);
    // Vizualization
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
    auto offsetReader = std::make_shared<io::DenFrame2DReader<float>>(ARG.fittedCoefficients[0]);
    auto aifframe = offsetReader->readFrame(ARG.ifz);
    float aifofset = aifframe->get(ARG.ifx, ARG.ify)
        + concentration->valueAt_intervalStart(ARG.ifx, ARG.ify, ARG.ifz);
    concentration->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(ARG.granularity, aif, convolutionMatrix);
    if(ARG.vizualize || !ARG.aifImageFile.empty() || !ARG.aifCsvFile.empty()
       || !ARG.basisImageFile.empty())
    {
        // See
        // https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
        if(!ARG.vizualize)
        {
            plt::backend("Agg");
        }
        if(ARG.showBasis || !ARG.basisImageFile.empty())
        {
            util::FourierSeries b(ARG.fittedCoefficients.size(), ARG.startTime, ARG.endTime, 1);
            if(ARG.showBasis)
            {
                b.plotFunctions();
            }
            if(!ARG.basisImageFile.empty())
            {
                b.storeFunctions(ARG.basisImageFile);
            }
        }
        plt::title(io::xprintf("Time attenuation curve, TST Fourier, x=%d, y=%d, z=%d.", ARG.ifx,
                               ARG.ify, ARG.ifz));
        plt::xlabel("Time [s]");
        if(ARG.water_value > 0)
        {
            plt::ylabel("Attenuation [HU]");
        } else
        {
            plt::ylabel("Attenuation");
        }
        std::vector<double> taxis_scatter;
        std::vector<double> plotme_scatter;
        if(ARG.staticReconstructionDir != "")
        {
            std::shared_ptr<util::ReconstructedSeriesEvaluator> _concentration
                = std::make_shared<util::ReconstructedSeriesEvaluator>(
                    ARG.staticReconstructionDir, ARG.sweepCount, ARG.sweepTime, ARG.sweepOffset);
            taxis_scatter = _concentration->nativeTimeDiscretization();
            plotme_scatter = _concentration->nativeValuesIn(ARG.ifx, ARG.ify, ARG.ifz);
            for(uint32_t i = 0; i != plotme_scatter.size(); i++)
            {
                taxis_scatter[i] = taxis_scatter[i] / ARG.secLength;
                if(ARG.water_value > 0)
                {
                    float v = plotme_scatter[i];
                    float hu = 1000 * (v / ARG.water_value - 1.0);
                    plotme_scatter[i] = hu;
                }
            }
        }
        std::vector<double> taxis;
        float* _taxis = new float[ARG.granularity];
        concentration->timeDiscretization(ARG.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            float v = aif[i] + aifofset;
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
        if(ARG.staticReconstructionDir != "")
        {
            std::map<std::string, std::string> pltargs;
            pltargs.insert(std::pair<std::string, std::string>("c", "Orange"));
            plt::scatter(taxis_scatter, plotme_scatter, 90.0, pltargs);
        }
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
            csv.writeLine(io::xprintf("perfviz-fourier generated AIF (x,y,z)=(%d, %d, %d) with "
                                      "baseSize %d, startTime=%f and endTime=%f",
                                      ARG.ifx, ARG.ify, ARG.ifz, baseSize, ARG.startTime,
                                      ARG.endTime));
            csv.writeVector(io::xprintf("time_harmonic%d_interp", baseSize), taxis);
            csv.writeVector(io::xprintf("value_harmonic%d_interp", baseSize), plotme);
            csv.close();
        }
        delete[] _taxis;
    }
    if(ARG.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    //    ARG.lambdaRel = 0.0;
    utils::TikhonovInverse ti(ARG.lambda_rel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, ARG.granularity);
    // Test what is the projection of convolutionMatrix to the last element of aif

    util::TimeSeriesDiscretizer tsd(concentration, ARG.dimx, ARG.dimy, ARG.dimz, ARG.startTime,
                                    ARG.endTime, ARG.secLength, ARG.threads);
    if(ARG.vizualize)
    {
        tsd.visualizeConvolutionKernel(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity,
                                       convolutionMatrix);
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", ARG.outputFolder.c_str()), ARG.dimx, ARG.dimy, ARG.dimz);
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
        LOGI << io::xprintf("CBF time is set as %f.", ARG.cbf_time);
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w,
                                       ARG.cbf_time);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
    PRG.endLog(true);
}
