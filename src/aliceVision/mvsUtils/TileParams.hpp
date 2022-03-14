// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/system/Logger.hpp>
#include <aliceVision/mvsData/ROI.hpp>
#include <vector>

namespace aliceVision {
namespace mvsUtils {

/**
 * @brief Tile Parameters
 */
struct TileParams
{
  // user parameters

  int width = -1;  // if < 0 no tile, use the entire image
  int height = -1; // if < 0 no tile, use the entire image
  int padding = 0;
};

 /**
 * @brief Get tile list from tile parameters and image width/height
 * @param[in] tileParams the tile parameters
 * @param[in] originalWidth the image original width
 * @param[in] originalHeight the image original height
 * @param[out] out_tileDepthMap the output tile ROI list
 */
void getTileList(const TileParams& tileParams, int originalWidth, int originalHeight, std::vector<ROI>& out_tileList);

} // namespace mvsUtils
} // namespace aliceVision
