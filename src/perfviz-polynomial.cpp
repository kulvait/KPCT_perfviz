// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "Frame2DReaderI.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/PolynomialSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#ifdef DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

using namespace CTL;

/// Arguments of the main function.
struct Arguments
{
    /// Function to parse function parameters.
    int parseArguments(int argc, char* argv[]);

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> fittedCoefficients;

    /// Number of threads
    uint16_t threads = 0;

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

	bool allowNegativeValues = false;

    bool chebyshev = false;
    bool legendre = false;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool vizualize = false;
    bool showBasis = false;
    bool showAIF = false;
    bool stopAfterVizualization = false;
    bool stopAfterTTP = false;
    std::string aifImageFile = "";
    std::string basisImageFile = "";
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Visualization of perfusion parameters CT based on Legendre fiting." };
    app.add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")->required();
    app.add_option("ify", ify, "Pixel based y coordinate of arthery input function")->required();
    app.add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")->required();
    app.add_option("output_folder", outputFolder,
                   "Folder to which output data after the linear regression. It needs to be "
                   "directory or symbol - to indicate that no computation should be performed.")
        ->required();
    app.add_option("fitted_coeficients", fittedCoefficients,
                   "Legendre coeficients fited by the algorithm. Orderred from the first "
                   "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    CLI::Option_group * pol_og = app.add_option_group("Polynomial type");
    pol_og->add_flag("--chebyshev", chebyshev, "Use Chebyshev polynomials.");
    pol_og->add_flag("--legendre", legendre, "Use Legendre polynomials.");
    pol_og->require_option(1);
    CLI::Option_group* interval_og = app.add_option_group(
        "Interval specification and basis functions specification.",
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
    app.add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    CLI::Option_group* flow_og = app.add_option_group("Program flow parameters");
    flow_og
        ->add_option("-j,--threads", threads,
                     "Number of extra threads that application can use. Defaults to 0 which means "
                     "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    flow_og->add_flag("--only-ttp", stopAfterTTP, "Compute only TTP.");
    CLI::Option_group* vizual_og
        = app.add_option_group("Vizualization configuration.",
                               "Configure output of basis and AIF to images and vizualization");
    vizual_og->add_flag("-v,--vizualize", vizualize, "Vizualization.");
    vizual_og->add_flag("--show-basis", showBasis, "Show basis.");
    vizual_og->add_flag("--show-aif", showAIF, "Show AIF.");

    try
    {
        std::string err;
        app.parse(argc, argv);
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
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            return -1;
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << io::xprintf("START %s", argv[0]);
    Arguments a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    uint16_t dimx, dimy, dimz;
    io::DenFileInfo di(a.fittedCoefficients[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
	util::polynomialType pt;
	if(	a.chebyshev)
	{
		pt = util::polynomialType::Chebyshev;
	}else if(a.legendre)
	{
		pt = util::polynomialType::Legendre;
	}
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::PolynomialSeriesEvaluator>(
            a.fittedCoefficients.size() - 1, a.fittedCoefficients, a.startTime, a.endTime, !a.allowNegativeValues, pt);
    // Vizualization
    float* convolutionMatrix = new float[a.granularity * a.granularity];
    float* aif = new float[a.granularity];
    concentration->timeSeriesIn(a.ifx, a.ify, a.ifz, a.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
    if(a.showBasis || !a.basisImageFile.empty())
    {
		std::shared_ptr<util::VectorFunctionI> b;
		if(a.chebyshev)
		{
			b = std::make_shared<util::ChebyshevPolynomialsExplicit>(a.fittedCoefficients.size() - 1, a.startTime, a.endTime, 1);
		}else
		{
			b = std::make_shared<util::LegendrePolynomialsExplicit>(a.fittedCoefficients.size() - 1, a.startTime, a.endTime, 1);
		}
        if(a.showBasis)
        {
            b->plotFunctions();
        }
        if(!a.basisImageFile.empty())
        {
            b->storeFunctions(a.basisImageFile);
        }
    }

    if(a.showAIF || !a.aifImageFile.empty())
    {
        std::vector<double> taxis;
        float* _taxis = new float[a.granularity];
        concentration->timeDiscretization(a.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != a.granularity; i++)
        {
            plotme.push_back(aif[i]);
            taxis.push_back(_taxis[i]);
        }
        plt::plot(taxis, plotme);
        if(a.showAIF)
        {
            plt::show();
        }
        if(!a.aifImageFile.empty())
        {
            plt::save(a.aifImageFile);
        }
        delete[] _taxis;
    }
    if(a.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, a.granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBF.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBV.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/MTT.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, a.startTime, a.endTime,
                                    a.secLength, a.threads);
    LOGD << "TTP computation.";
    tsd.computeTTP(a.granularity, ttp_w, aif);
    if(!a.stopAfterTTP)
    {
        LOGD << "CBV, CBF and MTT computation.";
        tsd.computePerfusionParameters(a.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
