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
class Args : public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> fittedCoefficients;

    /**
     * The first sweep and the last sweep should be identified with the ends of the interval.
     * startTime default is the end of the first sweep and endTime is the start of the last sweep
     * Data are from the experiments. Controls interval [ms].
     */
    float startTime = 4145, endTime = 43699;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    // Specify basis
    bool halfPeriodicFunctions = false;

    bool allowNegativeValues = false;

    // Tikhonov regularization parameter
    float lambdaRel = 0.2;

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
};

void Args::defineArguments()
{
    cliApp->add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")
        ->required();
    cliApp->add_option("ify", ify, "Pixel based y coordinate of arthery input function")
        ->required();
    cliApp->add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")
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
    CLI::Option_group* interval_og = cliApp->add_option_group(
        "Interval specification",
        "Specification of the time interval parameters and its discretization.");
    interval_og
        ->add_option("-i,--start-time", startTime,
                     "Start of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 4117, 247*16.6].")
        ->check(CLI::Range(0.0, 100000.0));
    interval_og
        ->add_option("-e,--end-time", endTime,
                     "End of the interval in miliseconds of the support of the functions of time "
                     "[defaults to 56000, duration of 9 sweeps].")
        ->check(CLI::Range(0.0, 100000.0));
    interval_og
        ->add_option("-c,--sec-length", secLength,
                     "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    interval_og
        ->add_option("-g,--granularity", granularity,
                     "Granularity of the time is number of time points to which time interval is "
                     "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));

    cliApp->add_flag("--half-periodic-functions", halfPeriodicFunctions,
                     "Use Fourier basis and include half periodic functions.");
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    cliApp->add_option("--lambda-rel", lambdaRel,
                       "Tikhonov regularization parameter, defaults to 0.2.");
    cliApp->add_flag("-v,--vizualize", vizualize, "Vizualize AIF and the basis.");
    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    flow_og->add_flag("--only-ttp", stopAfterTTP, "Compute only TTP.");
    CLI::Option_group* vizual_og
        = cliApp->add_option_group("Vizualization configuration.",
                                   "Configure output of basis and AIF to images and vizualization");
    vizual_og->add_flag("-v,--vizualize", vizualize, "Vizualization.");
    vizual_og->add_option("--water-value", water_value,
                   "If the AIF vizualization should be in HU, use this water_value.");
    vizual_og->add_flag("--show-basis", showBasis, "Show basis.");
    vizual_og->add_flag("--show-aif", showAIF, "Show AIF.");
    vizual_og->add_option("--store-aif", aifImageFile, "Store AIF into image file.");
    vizual_og->add_option("--store-basis", basisImageFile, "Store basis into image file.");
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
            plotme.push_back(aif[i]);
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
