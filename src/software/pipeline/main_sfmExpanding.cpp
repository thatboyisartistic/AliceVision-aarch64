// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>

#include <aliceVision/system/Timer.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/cmdline/cmdline.hpp>

#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <aliceVision/sfm/pipeline/relativePoses.hpp>
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/stl/mapUtils.hpp>

#include <aliceVision/track/tracksUtils.hpp>
#include <aliceVision/track/trackIO.hpp>

#include <aliceVision/sfm/liealgebra.hpp>

#include <aliceVision/multiview/triangulation/triangulationDLT.hpp>

#include <aliceVision/camera/camera.hpp>

#include <aliceVision/sfm/BundleAdjustmentCeres.hpp>
#include <aliceVision/sfm/BundleAdjustmentSymbolicCeres.hpp>

#include <cstdlib>
#include <random>
#include <regex>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace aliceVision::track;
using namespace aliceVision::sfm;

std::vector<boost::json::value> readJsons(std::istream& is, boost::json::error_code& ec)
{
    std::vector<boost::json::value> jvs;
    boost::json::stream_parser p;
    std::string line;
    std::size_t n = 0;


    while(true)
    {
        if(n == line.size())
        {
            if(!std::getline(is, line))
            {
                break;
            }

            n = 0;
        }

        //Consume at least part of the line
        n += p.write_some( line.data() + n, line.size() - n, ec);

        //If the parser found a value, add it
        if (p.done())
        {
            jvs.push_back(p.release());
            p.reset();
        }
    }

    if (!p.done())
    {
        //Try to extract the end
        p.finish(ec);
        if (ec.failed())
        {
            return jvs;
        }

        jvs.push_back(p.release());
    }

    return jvs;
}

double computeScore(const feature::FeaturesPerView & featuresPerView, const track::TracksMap & tracksMap, const std::vector<IndexT> & usedTracks, const IndexT viewId, const size_t maxLevel)
{
    const feature::MapFeaturesPerDesc& featuresPerDesc = featuresPerView.getFeaturesPerDesc(viewId);

    std::vector<std::set<std::pair<unsigned int, unsigned int>>> uniques(maxLevel - 1);

    for (auto trackId : usedTracks)
    {
        auto & track = tracksMap.at(trackId);

        const feature::PointFeatures& features = featuresPerDesc.at(track.descType);
        
        const IndexT featureId = track.featPerView.at(viewId);
        const Vec2 pt = features[featureId].coords().cast<double>();

        unsigned int ptx = (unsigned int)(pt.x());
        unsigned int pty = (unsigned int)(pt.y());

        for (unsigned int shift = 1; shift < maxLevel; shift++)
        {
            unsigned int lptx = ptx >> shift;
            unsigned int lpty = pty >> shift;

            uniques[shift - 1].insert(std::make_pair(lptx, lpty));
        }
    } 

    double sum = 0.0;
    for (unsigned int shift = 1; shift < maxLevel; shift++)
    {
        int size = uniques[shift - 1].size();
        if (size <= 1)
        {
            continue;
        }

        double w = pow(2.0, maxLevel - shift);
        sum += w * double(size);
    }

    return sum;
}


int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::vector<std::string> featuresFolders;
    std::string outputSfM;
    std::string tracksFilename;
    std::string pairsDirectory;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(), "SfMData file.")
        ("output,o", po::value<std::string>(&outputSfM)->required(), "Path to the output SfMData file.")
        ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(), "Path to folder(s) containing the extracted features.")
        ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),feature::EImageDescriberType_informations().c_str())
        ("tracksFilename,t", po::value<std::string>(&tracksFilename)->required(), "Tracks file.")
        ("pairs,p", po::value<std::string>(&pairsDirectory)->required(), "Path to the pairs directory.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("randomSeed", po::value<int>(&randomSeed)->default_value(randomSeed), "This seed value will generate a sequence using a linear random generator. Set -1 to use a random seed.");

    CmdLine cmdline("AliceVision Sfm Expanding");

    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set maxThreads
    HardwareContext hwc = cmdline.getHardwareContext();
    omp_set_num_threads(hwc.getMaxThreads());

    std::mt19937 randomNumberGenerator(randomSeed);

    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }

    // get imageDescriber type
    const std::vector<feature::EImageDescriberType> describerTypes =
        feature::EImageDescriberType_stringToEnums(describerTypesName);

    // features reading
    feature::FeaturesPerView featuresPerView;
    if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
    {
        ALICEVISION_LOG_ERROR("Invalid features.");
        return EXIT_FAILURE;
    }

    // Load tracks
    ALICEVISION_LOG_INFO("Load tracks");
    std::ifstream tracksFile(tracksFilename);
    if(tracksFile.is_open() == false)
    {
        ALICEVISION_LOG_ERROR("The input tracks file '" + tracksFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }
    std::stringstream buffer;
    buffer << tracksFile.rdbuf();
    boost::json::value jv = boost::json::parse(buffer.str());
    track::TracksMap mapTracks(track::flat_map_value_to<track::Track>(jv));

    // Compute tracks per view
    /*ALICEVISION_LOG_INFO("Estimate tracks per view");
    track::TracksPerView mapTracksPerView;
    for(const auto& viewIt : sfmData.views)
    {
        // create an entry in the map
        mapTracksPerView[viewIt.first];
    }
    track::computeTracksPerView(mapTracks, mapTracksPerView);*/


    //Result of pair estimations are stored in multiple files
    /*std::vector<sfm::ReconstructedPair> reconstructedPairs;
    const std::regex regex("pairs\\_[0-9]+\\.json");
    for(fs::directory_entry & file : boost::make_iterator_range(fs::directory_iterator(pairsDirectory), {}))
    {
        if (!std::regex_search(file.path().string(), regex))
        {
            continue;
        }

        std::ifstream inputfile(file.path().string());        

        boost::json::error_code ec;
        std::vector<boost::json::value> values = readJsons(inputfile, ec);
        for (const boost::json::value & value : values)
        {
            std::vector<sfm::ReconstructedPair> localVector = boost::json::value_to<std::vector<sfm::ReconstructedPair>>(value);
            reconstructedPairs.insert(reconstructedPairs.end(), localVector.begin(), localVector.end());
        }
    }*/

    {
        sfmData::SfMData sfmDataGT;
        sfmData::SfMData sfmDataEst;

        auto phIntrinsicGT = camera::createIntrinsic(camera::PINHOLE_CAMERA_RADIAL3, 1920, 1080, 980, 980, 50, 80);
        auto phPinholeGT = std::dynamic_pointer_cast<camera::PinholeRadialK3>(phIntrinsicGT);
        std::vector<double> paramsGT = {0.01, -0.01, 0.02};
        phPinholeGT->setDistortionParams(paramsGT);
        sfmDataGT.getIntrinsics()[0] = phPinholeGT;

        auto phIntrinsicEst = camera::createIntrinsic(camera::PINHOLE_CAMERA_RADIAL3, 1920, 1080, 950, 950, 50, 80);
        auto phPinholeEst = std::dynamic_pointer_cast<camera::PinholeRadialK3>(phIntrinsicEst);
        std::vector<double> paramsEst = {0.01, -0.01, 0.02};
        phPinholeEst->setDistortionParams(paramsEst);
        sfmDataEst.getIntrinsics()[0] = phPinholeEst;

        Vec3 direction = {1.0, 1.0, 1.0};
        direction = direction.normalized();
        Vec3 axis = {1.0, 1.0, 0.0};
        axis = axis.normalized();

        for (int i = 0; i < 40; i++)
        {
            Vec3 pos = direction * double(i) / 40.0;
            Eigen::Matrix3d R = SO3::expm(axis * double(0) * M_PI / (8*40.0));
            geometry::Pose3 poseGT(R, pos);
            sfmData::CameraPose cposeGT(poseGT);
            sfmDataGT.getPoses()[i] = cposeGT;
            sfmDataGT.getViews()[i] = std::make_shared<sfmData::View>("", i, 0, i, 1920, 1080);

            Eigen::Matrix3d Rup = SO3::expm(Vec3::Random() * ((i == 0)?0.0:0.05));
            Eigen::Vector3d tup = Vec3::Random() * ((i == 0)?0.0:0.1);

            geometry::Pose3 poseEst(Rup * R, pos + tup);
            sfmData::CameraPose cposeEst(poseEst, (i==0));
            sfmDataEst.getPoses()[i] = cposeEst;
            sfmDataEst.getViews()[i] = std::make_shared<sfmData::View>("", i, 0, i, 1920, 1080);

        }

        int tid = 0;

       /* for (double y = -2.0; y < 2.1; y+=0.1)
        {
            for (double x = -2.0; x < 2.1; x+=0.1)
            {
                sfmData::Landmark lGT;
                lGT.X = Vec3(x, y, 2.0 + std::abs(x));
                lGT.descType = feature::EImageDescriberType::SIFT;
                sfmDataGT.getLandmarks()[tid] = lGT;

                sfmData::Landmark lEst = lGT;
                lEst.X += Vec3::Random() * 0.9;
                sfmDataEst.getLandmarks()[tid] = lEst;

                tid++;
            }
        }*/


        /*for (double y = -2.0; y < 2.1; y+=0.1)
        {
            for (double x = -2.0; x < 2.1; x+=0.1)
            {
                sfmData::Landmark lGT;
                lGT.X = Vec3(x, y, 2.0);
                lGT.descType = feature::EImageDescriberType::SIFT;
                sfmDataGT.getLandmarks()[tid] = lGT;

                sfmData::Landmark lEst = lGT;
                lEst.X += Vec3::Random() * 0.9;
                sfmDataEst.getLandmarks()[tid] = lEst;

                tid++;
            }
        }*/


        for (double y = -2.0; y < 2.0; y+=0.1)
        {
            for (double x = -2.0; x < 2.0; x+=0.1)
            {
                sfmData::Landmark lGT;
                lGT.X = Vec3(x, y, 4.0);
                lGT.descType = feature::EImageDescriberType::SIFT;
                sfmDataGT.getLandmarks()[tid] = lGT;

                sfmData::Landmark lEst = lGT;
                lEst.X += Vec3::Random() * 0.9;
                sfmDataEst.getLandmarks()[tid] = lEst;

                tid++;
            }
        }

        /*for (double y = -1.0; y < 1.0; y+=0.1)
        {
            for (double x = -1.0; x < 1.0; x+=0.1)
            {
                sfmData::Landmark lGT;
                lGT.X = Vec3(x, y, 1.0) * 10000.0;
                lGT.descType = feature::EImageDescriberType::SIFT;
                sfmDataGT.getLandmarks()[tid] = lGT;

                sfmData::Landmark lEst = lGT;
                sfmDataEst.getLandmarks()[tid] = lEst;

                tid++;
            }
        }*/
        

        for (auto & pl : sfmDataGT.getLandmarks())
        {
            sfmData::Landmark & lEst = sfmDataEst.getLandmarks()[pl.first];
            
            for (auto & pp : sfmDataGT.getPoses())
            {
                sfmData::Observation obs;
                obs.x = phIntrinsicGT->project(pp.second.getTransform(), pl.second.X.homogeneous(), true);
                obs.scale = 1.0;
                obs.id_feat = pl.first;

                
                if (pp.second.getTransform()(pl.second.X)(2) < 0.1)
                {
                    std::cout << "removed" << std::endl;
                    continue;
                }

                pl.second.observations[pp.first] = obs;
                lEst.observations[pp.first] = obs;
            }
        }

        BundleAdjustmentSymbolicCeres::CeresOptions options;
        BundleAdjustment::ERefineOptions refineOptions = BundleAdjustment::REFINE_ROTATION | BundleAdjustment::REFINE_TRANSLATION |BundleAdjustment::REFINE_STRUCTURE | BundleAdjustment::REFINE_INTRINSICS_FOCAL | BundleAdjustment::REFINE_INTRINSICS_OPTICALOFFSET_ALWAYS;
        options.summary = true;

        BundleAdjustmentSymbolicCeres BA(options, 3);
        const bool success = BA.adjust(sfmDataEst, refineOptions);

    }

    

    return EXIT_SUCCESS;
}
