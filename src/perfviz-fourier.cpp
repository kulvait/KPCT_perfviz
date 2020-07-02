// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/FourierSeries.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "matplotlibcpp.h"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/FourierSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

namespace plt = matplotlibcpp;

using namespace CTL;
using namespace CTL::util;

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
    uint16_t ifx, ify, ifz;

    // Specify basis
    bool halfPeriodicFunctions = false;

    bool allowNegativeValues = false;

    // Tikhonov regularization parameter
    float lambdaRel = 0.2;
};

void Args::defineArguments()
{
    cliApp->add_option("ifx", ifx, "Voxel x coordinate of the arthery input function.")
        ->required();
    cliApp->add_option("ify", ify, "Voxel y coordinate of the arthery input function.")
        ->required();
    cliApp->add_option("ifz", ifz, "Voxel z coordinate of the arthery input function.")
        ->required();
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
    addVizualizationArgs();
    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    cliApp->add_flag("--half-periodic-functions", halfPeriodicFunctions,
                     "Use Fourier basis and include half periodic functions.");
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    cliApp->add_option("--lambda-rel", lambdaRel,
                       "Tikhonov regularization parameter, defaults to 0.2.");
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
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Visualization of perfusion parameters based on Fourier coefficients.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    uint16_t dimx, dimy, dimz;
    io::DenFileInfo di(ARG.fittedCoefficients[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    LOGI << io::xprintf("Start time is %f and end time is %f", ARG.startTime, ARG.endTime);
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::FourierSeriesEvaluator>(ARG.fittedCoefficients, ARG.startTime,
                                                         ARG.endTime, !ARG.allowNegativeValues,
                                                         ARG.halfPeriodicFunctions);
    // Vizualization
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
    concentration->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(ARG.granularity, aif, convolutionMatrix);
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
    if(ARG.showAIF || !ARG.aifImageFile.empty())
    {
        std::vector<double> taxis;
        float* _taxis = new float[ARG.granularity];
        concentration->timeDiscretization(ARG.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            if(ARG.water_value > 0)
            {
                plotme.push_back(aif[i] * 1000 / ARG.water_value);
            } else
            {
                plotme.push_back(aif[i]);
            }
            taxis.push_back(_taxis[i]);
        }
        plt::plot(taxis, plotme);
        if(ARG.showAIF)
        {
            plt::show();
        }
        if(!ARG.aifImageFile.empty())
        {
            plt::save(ARG.aifImageFile);
        }
        delete[] _taxis;
    }
    if(ARG.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    //    ARG.lambdaRel = 0.0;
    utils::TikhonovInverse ti(ARG.lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, ARG.granularity);
    // Test what is the projection of convolutionMatrix to the last element of aif

    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, ARG.startTime, ARG.endTime,
                                    ARG.secLength, ARG.threads);
    if(ARG.vizualize)
    {
        tsd.visualizeConvolutionKernel(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity,
                                       convolutionMatrix);
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    LOGD << "TTP computation.";
    tsd.computeTTP(ARG.granularity, ttp_w, aif);
    if(!ARG.stopAfterTTP)
    {
        LOGD << "CBV, CBF and MTT computation.";
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBF.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBV.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/MTT.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
    PRG.endLog(true);
}
