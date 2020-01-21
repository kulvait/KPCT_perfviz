// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/CTEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#if DEBUG
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
    std::string outputFile;

    /// Static reconstructions in a DEN format to use for linear regression.
    std::vector<std::string> coefficientVolumeFiles;

    /// Additional information about time offsets and other data
    std::vector<std::string> tickFiles;

    /// Number of threads
    uint16_t threads = 0;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    // Length of one second in the units of the domain
    float secLength = 1.0;

    uint32_t baseSize;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool vizualize = false;
    /**
     * @brief File to store AIF.
     */
    std::string storeAIF = "";
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Fast computation of parameter to locate AIF." };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    app.add_flag("-v,--vizualize", vizualize, "Vizualize AIF and the basis.");
    app.add_option("--store-aif", storeAIF, "Store AIF into image file.");

    app.add_option("output_file", outputFile, "File KVA coefs.")->required();
    app.add_option("static_reconstructions", coefficientVolumeFiles,
                   "Coeficients of the basis functions fited by the algorithm. Orderred in the "
                   "same order as the basis is sampled in a DEN file.")
        ->required()
        ->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
        if(coefficientVolumeFiles.size() < 2)
        {
            std::string err
                = io::xprintf("Small number of input files %d.", coefficientVolumeFiles.size());
            LOGE << err;
            io::throwerr(err);
        }
        for(std::string f : coefficientVolumeFiles)
        {
            std::string tickFile = f.substr(0, f.find_last_of(".")) + ".tick";
            if(!io::isRegularFile(tickFile))
            {
                std::string err = io::xprintf("Tick file %s does not exists.", tickFile.c_str());
                LOGE << err;
                throw std::runtime_error(err);
            } else
            {
                tickFiles.push_back(tickFile);
            }
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
    io::DenFileInfo di(a.coefficientVolumeFiles[0]);
    uint16_t dimx = di.dimx();
    uint16_t dimy = di.dimy();
    uint16_t dimz = di.dimz();
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::CTEvaluator>(a.coefficientVolumeFiles, a.tickFiles);
    uint32_t n = a.coefficientVolumeFiles.size();
    // Vizualization
    float* convolutionMatrix = new float[n * n];
    float* aif = new float[n]();
    aif[0] = 0.5;
    aif[1] = 1.0;
    aif[2] = 0.5;
    utils::TikhonovInverse::precomputeConvolutionMatrix(n, aif, convolutionMatrix);
    if(a.vizualize || !a.storeAIF.empty())
    {
        std::vector<double> taxis;
        float* _taxis = new float[n];
        concentration->timeDiscretization(n, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != n; i++)
        {
            plotme.push_back(aif[i]);
            taxis.push_back(_taxis[i]);
        }
        plt::plot(taxis, plotme);
        std::shared_ptr<util::CTEvaluator> conct
            = std::dynamic_pointer_cast<util::CTEvaluator>(concentration);
        plt::plot(taxis, plotme);
        std::vector<double> taxis_scatter = conct->nativeTimeDiscretization(a.ifz);
        std::vector<double> plotme_scatter = conct->nativeValuesIn(a.ifx, a.ify, a.ifz);
        plt::plot(taxis_scatter, plotme_scatter);
        if(a.vizualize)
        {
            plt::show();
        }
        if(!a.storeAIF.empty())
        {
            plt::save(a.storeAIF);
        }
        delete[] _taxis;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.075;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, n);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> kva
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.outputFile, dimx, dimy, dimz);
/*
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> kva
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>("/tmp/KVA.den", dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> kvb
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>("/tmp/KVB.den", dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> kvc
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>("/tmp/KVC.den", dimx, dimy, dimz);
*/
    std::shared_ptr<io::Frame2DReaderI<float>> startData
        = std::make_shared<io::DenFrame2DReader<float>>(a.tickFiles[0]);
    std::shared_ptr<io::Frame2DReaderI<float>> endData
        = std::make_shared<io::DenFrame2DReader<float>>(a.tickFiles[a.tickFiles.size() - 1]);
    std::shared_ptr<io::Frame2DI<float>> fs, fe;
    fs = startData->readFrame(0);
    fe = endData->readFrame(0);
    // Fifth element of frame is time
    float intervalStart, intervalEnd;
    intervalStart = fs->get(4, 0);
    intervalEnd = fe->get(4, 0);
    float start, end;
    for(std::size_t z = 0; z != dimz; z++)
    {
        fs = startData->readFrame(z);
        fe = endData->readFrame(z);
        start = fs->get(4, 0);
        end = fe->get(4, 0);
        if(start > intervalStart)
        {
            intervalStart = start;
        }
        if(end < intervalEnd)
        {
            intervalEnd = end;
        }
    }
    LOGD << io::xprintf("Computed interval start is %f and interval end is %f and secLength is %f",
                        intervalStart, intervalEnd, a.secLength);
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, intervalStart, intervalEnd,
                                    a.secLength, a.threads);
    LOGD << "KVA computation.";
    float* values = new float[dimx * dimy * n];
    // float* convol = new float[dimx * dimy * granularity]();
    float convol_i;
    for(int z = 0; z != dimz; z++)
    {
        concentration->frameTimeSeries(z, n, values);
        io::BufferedFrame2D<float> kvafr(std::numeric_limits<float>::lowest(), dimx, dimy);
        io::BufferedFrame2D<float> kvbfr(float(0), dimx, dimy);
        io::BufferedFrame2D<float> kvcfr(float(0), dimx, dimy);
        for(int x = 0; x != dimx; x++)
        {
            for(int y = 0; y != dimy; y++)
            {
                for(uint32_t i = 0; i != n; i++)
                {
                    convol_i = 0.0;
                    for(uint32_t j = 0; j != n; j++)
                    {
                        convol_i += convolutionMatrix[i * n + j]
                            * values[j * dimx * dimy + y * dimx + x];
                    }
                    if(convol_i * float(n - i - 1) / float(n - 1) > kvafr.get(x, y))
                    {
                        kvafr.set(convol_i * float(n - i - 1) / float(n - 1), x, y);
                    }
                    kvbfr.set(convol_i + kvbfr.get(x, y), x, y);
                    kvcfr.set(values[i * dimx * dimy + y * dimx + x] + kvcfr.get(x, y), x, y);
                }
            }
        }
        kva->writeFrame(kvafr, z);
    //    kvb->writeFrame(kvbfr, z);
    //    kvc->writeFrame(kvcfr, z);
    }
    delete[] values;
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
