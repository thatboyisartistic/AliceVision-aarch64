// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfm/sfm.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>
#include <aliceVision/track/TracksBuilder.hpp>
#include <aliceVision/track/trackIO.hpp>
#include <aliceVision/sfm/BundleAdjustment.hpp>


#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace aliceVision::track;
using namespace aliceVision::sfm;


int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::vector<std::string> featuresFolders;
    std::string tracksFilename;
    std::string outputSfM;

    // user optional parameters
    std::string outputSfMViewsAndPoses;
    std::string extraInfoFolder;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    std::pair<std::string, std::string> initialPairString("", "");

    sfm::ReconstructionEngine_sequentialSfM::Params sfmParams;
    bool lockScenePreviouslyReconstructed = true;
    int maxNbMatches = 0;
    int minNbMatches = 0;
    bool useOnlyMatchesFromInputFolder = false;
    bool computeStructureColor = true;

    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilename)->required(), "SfMData file.")
    ("tracksFilename,i", po::value<std::string>(&tracksFilename)->required(), "Tracks file.")
    ("output,o", po::value<std::string>(&outputSfM)->required(), "Path to the output SfMData file.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
    ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(), "Path to folder(s) containing the extracted features.")
    ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),feature::EImageDescriberType_informations().c_str())
    ("minAngleInitialPair", po::value<float>(&sfmParams.minAngleInitialPair)->default_value(sfmParams.minAngleInitialPair),"Minimum angle for the initial pair.")
    ("maxAngleInitialPair", po::value<float>(&sfmParams.maxAngleInitialPair)->default_value(sfmParams.maxAngleInitialPair), "Maximum angle for the initial pair.");

    CmdLine cmdline("AliceVision bootstrapSfM");

    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set maxThreads
    HardwareContext hwc = cmdline.getHardwareContext();
    omp_set_num_threads(hwc.getMaxThreads());

    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }

    std::ifstream tracksFile(tracksFilename);
    if (tracksFile.is_open() == false)
    {
        ALICEVISION_LOG_ERROR("The input tracks file '" + tracksFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }
    std::stringstream buffer;
    buffer << tracksFile.rdbuf();
    boost::json::value jv = boost::json::parse(buffer.str());
    track::TracksMap mapTracks(flat_map_value_to<Track>(jv));
    


    std::map<Pair, unsigned int> covisibility;
    for (const auto & item : mapTracks)
    {
        const auto & track = item.second;

        for (auto it = track.featPerView.begin(); it != track.featPerView.end(); it++)
        {
            Pair p;
            p.first = it->first;

            for (auto next = std::next(it); next != track.featPerView.end(); next++)
            {
                p.second = next->first;
                
                if (covisibility.find(p) == covisibility.end())
                {
                    covisibility[p] = 0;
                }
                else
                {
                    covisibility[p]++;
                }
            }
        }
    }

    int count = 0;
    for (auto item : covisibility)
    {
        if (item.second > 10)
        {
            count++;
        }
    }

    std::cout << count << std::endl;

    return EXIT_SUCCESS;
}
